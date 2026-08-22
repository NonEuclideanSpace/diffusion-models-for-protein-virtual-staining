from collections import Counter

import pytest

from pvs.data.hpa import Gene,cells_per_group,matched_controls,summarize

TSV="/home/claude/data/hpa/subcellular_location.tsv"


def _gene(name,reliability="Approved",location="Nucleoplasm",spatial="",intensity=""):
    return Gene(f"ENSG{name}",name,reliability,location,spatial,intensity)


def test_gene_properties():
    plain=_gene("A")
    assert not plain.is_spatially_variable and not plain.multilocalizing
    assert plain.locations==("Nucleoplasm",)
    flagged=_gene("B",location="Nucleoplasm;Cytosol",spatial="Nucleoplasm")
    assert flagged.is_spatially_variable and flagged.multilocalizing
    assert flagged.locations==("Nucleoplasm","Cytosol")
    assert flagged.stratum==("Approved","Nucleoplasm;Cytosol")


def test_controls_match_stratum_and_exclude_positives():
    positives=[_gene("P1",spatial="x"),_gene("P2","Supported","Cytosol",spatial="x")]
    pool=[_gene(f"C{i}") for i in range(5)]+[_gene(f"D{i}","Supported","Cytosol") for i in range(5)]
    controls=matched_controls(positives+pool,positives,per_positive=2,seed=0)
    assert len(controls)==4
    assert {c.ensembl for c in controls}.isdisjoint({p.ensembl for p in positives})
    assert Counter(c.stratum for c in controls)==Counter({("Approved","Nucleoplasm"):2,
                                                          ("Supported","Cytosol"):2})


def test_controls_are_never_reused():
    positives=[_gene(f"P{i}",spatial="x") for i in range(3)]
    pool=[_gene(f"C{i}") for i in range(4)]
    controls=matched_controls(positives+pool,positives,per_positive=2,seed=0)
    assert len({c.ensembl for c in controls})==len(controls)<=4


def test_controls_are_deterministic_given_a_seed():
    positives=[_gene("P",spatial="x")]
    pool=[_gene(f"C{i}") for i in range(20)]
    first=matched_controls(positives+pool,positives,2,seed=7)
    second=matched_controls(positives+pool,positives,2,seed=7)
    assert [g.ensembl for g in first]==[g.ensembl for g in second]


def test_cells_per_group_counts_by_gene_and_line():
    rows=[{"gene_names":"A","atlas_name":"U2OS"}]*3+[{"gene_names":"A","atlas_name":"HEK293"}]
    counts=cells_per_group(rows)
    assert counts[("A","U2OS")]==3 and counts[("A","HEK293")]==1


def test_summary_shape():
    s=summarize([_gene("A"),_gene("B",location="Nucleoplasm;Cytosol")])
    assert s["count"]==2 and s["multilocalizing"]==1
    assert s["reliability"]=={"Approved":2}


@pytest.mark.skipif(not __import__("pathlib").Path(TSV).exists(),reason="HPA metadata not fetched")
def test_probe_a_selection_on_the_real_metadata():
    """Controls must match on main location too, otherwise a positive result would only be
    saying that multi-located proteins get higher entropy, which proves nothing about
    cell-to-cell variation."""
    from pvs.data.hpa import probe_a_selection
    selection=probe_a_selection(TSV,per_positive=2)
    positive,control=summarize(selection["positive"]),summarize(selection["control"])
    assert positive["count"]==186
    assert control["count"]>2*positive["count"]*0.8
    positive_rate=positive["multilocalizing"]/positive["count"]
    control_rate=control["multilocalizing"]/control["count"]
    assert abs(positive_rate-control_rate)<0.10


def test_controls_match_cell_line_when_it_is_supplied():
    """Omitting this invalidated a real run: round suspension lines fell almost entirely among
    controls and manufactured a 5.5x difference in apparent heterogeneity."""
    positives=[_gene("P1",spatial="x"),_gene("P2",spatial="x")]
    pool=[_gene(f"A{i}") for i in range(4)]+[_gene(f"B{i}") for i in range(4)]
    lines={**{f"A{i}":"U2OS" for i in range(4)},**{f"B{i}":"HEL" for i in range(4)},
           "P1":"U2OS","P2":"U2OS"}
    controls=matched_controls(positives+pool,positives,per_positive=2,seed=0,cell_lines=lines)
    assert controls
    assert all(lines[c.name]=="U2OS" for c in controls),"controls must share the line"


def test_composition_distance_detects_an_unbalanced_selection():
    from pvs.data.hpa import composition_distance
    a=[_gene("A1"),_gene("A2")]
    b=[_gene("B1"),_gene("B2")]
    same={"A1":"U2OS","A2":"U2OS","B1":"U2OS","B2":"U2OS"}
    different={"A1":"U2OS","A2":"U2OS","B1":"HEL","B2":"REH"}
    assert composition_distance(a,b,same)==pytest.approx(0.0)
    assert composition_distance(a,b,different)==pytest.approx(1.0)


def test_dominant_cell_lines_picks_the_mode():
    from pvs.data.hpa import dominant_cell_lines
    rows=[{"gene_names":"G","atlas_name":"U2OS"}]*3+[{"gene_names":"G","atlas_name":"HEL"}]
    assert dominant_cell_lines(rows)=={"G":"U2OS"}
