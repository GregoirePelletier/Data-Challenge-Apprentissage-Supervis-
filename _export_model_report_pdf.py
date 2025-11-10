# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 15:58:29 2025

@author: saout
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import mm
from sklearn.metrics import accuracy_score, f1_score
from typing import Union

def export_model_report_pdf(
    estimator,
    X_test,
    y_test,
    pdf_path: Union[str, Path],
    title: str = "Model Report"
):
    """
    - Calcule Accuracy et F1_weighted sur (X_test, y_test)
    - Affiche le nom du modèle + ses hyperparamètres (get_params)
    - Écrit un PDF compact à `pdf_path` (création auto du dossier)

    Retourne: (f1_weighted: float, accuracy: float, model_name: str, params_dict: dict)
    """
    # --- compat Path & création du dossier ---
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # --- métriques ---
    y_pred = estimator.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average="weighted")

    # --- styles & en-tête ---
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    normal = styles["BodyText"]
    heading = styles["Heading2"]

    elements = []
    elements.append(Paragraph(title, title_style))
    elements.append(Spacer(1, 6))

    model_name = estimator.__class__.__name__
    elements.append(Paragraph(f"<b>Modèle&nbsp;:</b> {model_name}", normal))
    elements.append(Paragraph(f"<b>Accuracy (test)&nbsp;:</b> {acc:.4f}", normal))
    elements.append(Paragraph(f"<b>F1_weighted (test)&nbsp;:</b> {f1w:.4f}", normal))
    elements.append(Spacer(1, 6))

    # --- paramètres du modèle ---
    params = estimator.get_params(deep=True)
    rows = [["Paramètre", "Valeur"]]
    for k, v in sorted(params.items(), key=lambda x: x[0]):
        val = repr(v)
        if len(val) > 120:
            val = val[:117] + "…"
        rows.append([k, val])

    table = Table(rows, colWidths=[60*mm, 110*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
    ]))
    elements.append(Paragraph("<b>Paramètres du modèle</b>", heading))
    elements.append(table)

    # --- génération PDF ---
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    doc.build(elements)

    return f1w, acc, model_name, params
