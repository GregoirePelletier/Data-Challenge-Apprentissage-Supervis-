# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 11:01:24 2025

@author: saout
"""



import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
import seaborn as sns


def spineplot_with_props(df, x, y, min_count=1, gap=0.01, return_fig=False):
    """
    Mosaic (spineplot-like) annoté avec P(Y=y_val | X=x_val).
    - min_count: n'affiche pas le pourcentage si le compte est trop petit.
    - return_fig: si True, renvoie la figure pour PdfPages.
    """
    d = df[[x, y]].dropna().copy()
    d[x] = d[x].astype(str)   # évite KeyError '0' vs 0
    d[y] = d[y].astype(str)

    ct = pd.crosstab(d[x], d[y])                   # comptes
    prop = ct.div(ct.sum(axis=1), axis=0).fillna(0)

    def labelizer(k):
        xv, yv = k
        if ct.loc[xv, yv] < min_count:
            return ""
        return f"{prop.loc[xv, yv]:.0%}"

    fig, _ = mosaic(ct.stack(), gap=gap, labelizer=labelizer)
    ax = plt.gca()
    ax.set_xlabel(x); ax.set_ylabel(y)
    ax.set_title(f"Mosaic {x} vs {y} — P({y}|{x})")
    fig.tight_layout()

    if return_fig:
        return fig



    

