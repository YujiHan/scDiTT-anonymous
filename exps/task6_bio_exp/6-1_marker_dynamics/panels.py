from __future__ import annotations

from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────
SYMBOL_SPACE: Dict[str, str] = {
    "ZB": "var_names",
    "DR": "var_names",
    "GA": "gene_symbols",
    "MA": "var_names",
    "SC": "var_names",
}

# ─────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────
ALIASES: Dict[str, Dict[str, str]] = {
    "ZB": {"tbxta": "ta", "gata1": "gata1a"},
}

# ─────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────
PANELS: Dict[str, Dict[str, List[str]]] = {
    # ── ZB —— 97 ─────────────────────────────────────────────────────────
    "ZB": {
        "maternal_pluripotency": ["pou5f3", "nanog", "sox2", "sox19b", "klf17",
                                  "sox19a", "vox", "tdgf1"],
        "pgc": ["nanos3", "dnd1", "ddx4"],
        "mesoderm": ["ta", "tbx16", "msgn1", "noto", "mespaa", "mespba", "tbx6",
                     "gsc", "wnt8a", "wnt11", "ved", "eve1", "cdx4", "ripply1",
                     "ripply2", "her1", "her7", "aldh1a2", "pcdh8", "snai1a"],
        "endoderm": ["sox17", "foxa2", "sox32", "gata5", "gata6", "foxa1",
                     "foxa3", "mixl1", "nkx2.7", "sebox"],
        "neural": ["sox3", "neurog1", "elavl3", "otx2", "pax6a", "pax2a", "olig2",
                   "her4.1", "her4.2", "nkx2.2a", "shha", "irx3a", "zic1", "zic2b",
                   "sox21a", "neurod1", "emx3", "six3b", "rx3"],
        "neural_crest": ["foxd3", "sox10", "tfap2a"],
        "hematopoietic_vascular": ["gata1a", "gata2a", "tal1", "lmo2", "runx3",
                                   "spi1b", "hbbe1.1", "hbae3", "hbbe3", "myb",
                                   "fli1a", "etv2", "drl"],
        "epidermis_periderm": ["krt4", "krt8", "krt18", "cldnb", "cldni", "tp63",
                               "gata3", "krt17", "icn", "anxa1a", "col1a1a", "epcam"],
        "muscle": ["myod1", "myog", "desma", "acta1b", "smyhc1"],
        "organ_primordia": ["pax8", "lhx1a", "wt1a", "cdh17"],
    },
    # ── DR —— 60 ─────────────────────────────────────────────────────────
    "DR": {
        "patterning_tf": ["Kr", "kni", "wg", "hh", "zen", "sog", "twi", "srp",
                          "Doc2", "pnr", "GATAe", "byn", "fkh", "trh"],
        "hox": ["Ubx", "abd-A", "Abd-B", "Antp", "Scr", "pb", "lab"],
        "neural_pns": ["elav", "Syt1", "nSyb", "repo", "ase", "grh", "ato", "sens",
                       "sv", "ct", "beat-Ia", "nompA", "pdm3", "Hey",
                       "E(spl)m8-HLH", "Dl", "Ser", "eya", "nub", "vvl", "ap", "kn"],
        "muscle": ["Mef2", "Mhc", "Act57B", "sls", "up", "TpnC41C", "wupA",
                   "Zasp52", "org-1", "tey", "sr"],
        "germline": ["vas", "nos", "pgc"],
        "salivary_gland": ["CrebA", "ng2", "PH4alphaSG1"],
    },
    # ── GA —— 43 ─────────────────────────────────────────────────────────
    "GA": {
        "early_germline_pluripotency": ["POU5F1", "DPPA3", "DPPA5"],
        "proliferation": ["MKI67", "TOP2A", "UBE2C"],
        "meiosis": ["DDX4", "SYCP1", "SYCP2", "SYCP3", "STRA8", "SPO11", "REC8",
                    "MAEL", "MEIOB", "DMC1", "HORMAD1", "ZGLP1", "ANHX",
                    "RAD51AP2", "SMC1B", "TEX12", "TEX15"],
        "oocyte_maturation": ["FIGLA", "NOBOX", "GDF9", "ZP2", "ZP3", "OOSP2",
                              "H1FOO", "NLRP7", "PADI6", "KHDC3L", "TUBB8",
                              "SOHLH1", "LHX8"],
        "male_germline": ["MAGEA4", "ID4"],
        "somatic_contamination_ctrl": ["COL1A1", "UPK3B", "KRT19", "HBA1", "PTPRC"],
    },
    # ── MA —— 86 ─────────────────────────────────────────────────────────
    "MA": {
        "multipotent_progenitor": ["PTF1A", "GATA4", "HES1", "CPM", "SPP1",
                                   "LGALS3", "ID1", "ID3"],
        "tip_acinar": ["CPA1", "CPA2", "CPB1", "PRSS1", "PRSS2", "CTRB1", "CTRB2",
                       "CELA2A", "CELA3A", "CLPS", "RBPJL", "GP2", "CTRC", "SEL1L"],
        "trunk_duct": ["KRT7", "CFTR", "TACSTD2", "CLDN10", "AQP1", "ST18"],
        "endocrine_progenitor": ["NEUROG3", "FEV", "NEUROD1", "PAX4", "NKX2-2",
                                 "INSM1", "RFX6", "ISL1", "ARX", "PCSK1", "PCSK2",
                                 "CHGA", "CHGB", "SCG2", "SCG3", "SCG5", "PCSK1N",
                                 "MYT1", "HES6", "BTBD17"],
        "hormone_terminal": ["INS", "GCG", "SST", "GHRL", "PPY", "IAPP", "MAFA",
                             "MAFB", "UCN3", "HHEX", "LEPR", "ETV1", "IRX1",
                             "IRX2", "DPP4", "SLC30A8", "ABCC8"],
        "hepatoblast_ehbd": ["AFP", "ALB", "APOA1", "APOA2", "APOB", "TTR",
                             "SERPINA1", "TBX3", "FGB", "TF", "RBP4"],
        "intestine": ["CDX2", "FABP1", "TFF3"],
        "proliferation": ["MKI67", "TOP2A", "CCNB1", "CDK1", "PCNA", "UBE2C", "BIRC5"],
    },
    # ── SC —— 106 ────────────────────────────────────────────────────────
    "SC": {
        "mef_fibroblast_down": ["Col1a1", "Col1a2", "Col3a1", "Thy1", "Postn",
                                "Fn1", "Vim", "Snai1", "Snai2", "Zeb1", "Zeb2",
                                "Twist1", "Acta2", "Tagln", "Lum", "Dcn",
                                "Serpine1", "Timp1", "Mmp2", "Mmp3", "Fbn1",
                                "Eln", "Pdgfrb", "S100a4"],
        "met_up": ["Cdh1", "Epcam", "Krt8", "Krt18", "Krt19", "Cldn3", "Cldn4",
                   "Cldn6", "Cldn7", "Ocln", "Rab25", "Dsp", "Perp"],
        "pluripotency_late_up": ["Pou5f1", "Nanog", "Sox2", "Esrrb", "Klf4",
                                 "Klf5", "Dppa5a", "Dppa3", "Utf1", "Sall4",
                                 "Lin28a", "Tdgf1", "Fbxo15", "Tfcp2l1", "Tet1",
                                 "Dnmt3l", "Lefty1", "Lefty2", "Zscan4d",
                                 "Zscan4c", "Morc1"],
        "trophoblast_offtarget": ["Cdx2", "Krt7", "Elf5", "Tfap2c", "Plac1",
                                  "Prl3d1", "Hand1", "Gata3", "Ascl2", "Peg3",
                                  "Peg10", "Psg17", "Tpbpa", "Esx1"],
        "neural_offtarget": ["Sox1", "Pax6", "Nes", "Tubb3", "Sox10", "Gfap",
                             "Aqp4", "Slc1a3", "Ascl1", "Neurod1", "Neurog2",
                             "Dcx", "Elavl3", "Cspg4", "Pdgfra"],
        "xen_endoderm_offtarget": ["Gata4", "Sox17", "Foxa2", "Ttr", "Apoa1",
                                   "Apoe", "Dab2", "Lama1", "Col4a1", "Sparc"],
        "stress_proliferation": ["Myc", "Top2a", "Ccnb1", "Cdkn1a", "Cdkn2a",
                                 "Il6", "Isg15", "Hspa1a", "Hspa1b"],
    },
}

# ─────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────
MISSES: Dict[str, List[str]] = {
    "ZB": ["tbxta", "ntla", "gata1", "runx1", "spi1a", "gata2b", "klf1", "ikzf1",
           "cmyb", "kdrl", "cdh5", "npas4l", "hbae1.1", "vent", "ctcf", "flh",
           "chrd", "bmp4", "bmp2b", "snai1b", "tbx24", "spt", "cas", "bon",
           "otx1", "gli1", "tubb5", "snai2", "crestin", "mylpfa", "tnnt2a",
           "tnnt3a", "myl1", "mki67", "pcna", "fabp11a", "wt1b", "nphs1",
           "nphs2", "slc20a1a", "ponzr3", "krtt1c19e"],
    "DR": ["hb", "gt", "tll", "hkb", "eve", "ftz", "run", "h", "odd", "prd",
           "slp1", "slp2", "opa", "en", "ptc", "ci", "Dfd", "croc", "sim", "vnd",
           "ind", "msh", "sna", "dpp", "tin", "bap", "bin", "Doc1", "Doc3", "ush",
           "hnt", "pros", "dpn", "nrv2", "brp", "futsch", "how", "Dg", "mid",
           "salm", "esg", "stg", "Traf4", "Ance", "C15", "Sox15", "so", "amos",
           "cato", "nerfin-1", "N", "osk", "tud", "aub", "piwi", "AGO3", "Sxl",
           "otu", "bam", "bgcn", "gcl", "Race", "egr", "sage", "Sgs1", "Sgs3",
           "Sgs4", "Sgs5", "ng1", "PH4alphaSG2", "Pasilla", "sano", "Act87E",
           "Act79B", "Tm1", "Tm2", "slou", "lb", "vg", "col", "nompC", "Rfabg",
           "22C10", "trio"],
    "GA": ["NANOG", "NANOS3", "TFAP2C", "PRDM1", "SOX17", "KIT", "LIN28A",
           "ALPL", "DND1", "KLF4", "TFCP2L1", "UTF1", "L1TD1", "ITGA6", "DAZL",
           "PIWIL2", "TEX14", "MSH5", "ZP1", "BMP15", "SOHLH2", "TAF4B",
           "NANOS2", "DNMT3L", "PIWIL4", "TSPAN33", "EGR4", "SIX1", "TCF3",
           "GAGE1", "DPPA4", "POU3F1", "SALL4", "CCNB1", "CDK1", "PCNA", "TYMS",
           "BIRC5", "AURKB", "FOXL2", "NR5A1", "AMH", "SOX9", "WT1", "GATA4",
           "INHA", "ARX", "PDGFRA", "DLK1", "TCF21"],
    "MA": ["PDX1", "SOX9", "NKX6-1", "GATA6", "HNF1B", "ONECUT1", "FOXA2",
           "MNX1", "PROX1", "NOTCH1", "NOTCH2", "JAG1", "TEAD1", "YAP1",
           "AMY2A", "PNLIP", "BHLHA15", "KRT19", "KRT8", "KRT18", "ANXA4",
           "MUC1", "SOX4", "TOX3", "SIX3", "G6PC2", "KCNJ11", "HNF4A", "KRT23",
           "LGALS2", "CLDN4", "CDX1", "VIL1", "FABP2", "APOA4", "LGALS4",
           "MUC2", "OLFM4", "LGR5"],
    "SC": ["Zfp42", "Obox6", "Klf2", "Dppa2", "Dppa4", "Lin28b", "Nr0b1", "Tcl1",
           "Prdm14", "Nr5a2", "Tet2", "Dnmt3b", "Gdf3", "Fgf4", "L1td1", "Trim71",
           "Twist2", "Esrp1", "Esrp2", "Grhl2", "Grhl3", "Ovol1", "Ovol2",
           "Tfap2a", "Map2", "Olig1", "Olig2", "S100b", "Mbp", "Plp1", "Gata6",
           "Sox7", "Hnf4a", "Afp", "Amn", "Cubn", "Mycn", "Mki67", "Trp53",
           "Ifitm3"],
}

DATA_DRIVEN_FALLBACK_THRESHOLD = 8
DATA_DRIVEN_K = 30

USABLE_COMBOS: List[Tuple[str, str]] = [
    ("ZB", "interpolation"),
    ("DR", "interpolation"), ("DR", "extrapolation"),
    ("GA", "interpolation"), ("GA", "extrapolation"),
    ("SC", "interpolation"), ("SC", "extrapolation"),
]
EXCLUDED_COMBOS: List[Tuple[str, str]] = [
    ("ZB", "extrapolation"), ("MA", "interpolation"), ("MA", "extrapolation"),
]
ALL_COMBOS = USABLE_COMBOS + EXCLUDED_COMBOS

ARMS_10: List[str] = [
    "neg_mean_noise", "neg_shuffle",
    "PI-SDE", "scIMF", "scNODE", "MIOFlow", "PRESCIENT", "scDiffEq",
    "scDiTT", "pos_split_half",
]

GATE_K_SE = 3.0
GATE_ABS_FLOOR = 0.02
SHAPE_TOL_SE = 2.0

FROZEN_INFORMATIVE = {
    ("ZB", "interpolation"): 65, ("ZB", "extrapolation"): 16,
    ("DR", "interpolation"): 52, ("DR", "extrapolation"): 26,
    ("GA", "interpolation"): 29, ("GA", "extrapolation"): 38,
    ("MA", "interpolation"): 66,
    ("SC", "interpolation"): 74, ("SC", "extrapolation"): 78,
}
FROZEN_SHAPES = {
    ("ZB", "interpolation"): (23, 24, 18, 0), ("ZB", "extrapolation"): (12, 4, 0, 0),
    ("DR", "interpolation"): (19, 20, 12, 1), ("DR", "extrapolation"): (10, 3, 12, 1),
    ("GA", "interpolation"): (22, 4, 3, 0), ("GA", "extrapolation"): (7, 6, 25, 0),
    ("MA", "interpolation"): (50, 16, 0, 0),
    ("SC", "interpolation"): (37, 2, 35, 0), ("SC", "extrapolation"): (7, 3, 68, 0),
}


def panel_genes(ds: str) -> List[str]:
    out, seen = [], set()
    for genes in PANELS[ds.upper()].values():
        for g in genes:
            if g not in seen:
                seen.add(g)
                out.append(g)
    return out


def gene_lineage(ds: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for lin, genes in PANELS[ds.upper()].items():
        for g in genes:
            out.setdefault(g, lin)
    return out


def self_check() -> None:
    want_panel = {"ZB": 97, "DR": 60, "GA": 43, "MA": 86, "SC": 106}
    want_miss = {"ZB": 42, "DR": 84, "GA": 50, "MA": 39, "SC": 40}
    for ds, n in want_panel.items():
        got = len(panel_genes(ds))
        assert got == n, f"{ds} panel {got} != frozen value {n}"
    for ds, n in want_miss.items():
        got = len(MISSES[ds])
        assert got == n, f"{ds} dropped {got} != frozen value {n}"
    assert sum(want_panel.values()) == 392
    assert sum(want_miss.values()) == 255
    for ds in want_panel:
        flat = [g for gs in PANELS[ds].values() for g in gs]
        dup = len(flat) - len(set(flat))
        assert dup == 0, f"{ds} panel contains {dup} duplicate genes, inconsistent with the frozen count convention"


if __name__ == "__main__":
    self_check()
    for ds in ("ZB", "DR", "GA", "MA", "SC"):
        print(f"{ds}: panel={len(panel_genes(ds))}  miss={len(MISSES[ds])}  "
              f"space={SYMBOL_SPACE[ds]}  aliases={ALIASES.get(ds, {})}")
    print("total panel =", sum(len(panel_genes(d)) for d in PANELS))
    print("total miss  =", sum(len(v) for v in MISSES.values()))
    print("self_check PASSED")
