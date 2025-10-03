# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 15:58:29 2025

@author: saout
"""

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import mm

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score



def export_model_report_pdf(estimator, X_test, y_test, pdf_path: str, title: str = "Model Report"):
    """
    - Calcule l'accuracy sur X_test,y_test
    - Extrait le nom du modèle + ses paramètres (get_params)
    - Écrit un PDF compact à pdf_path
    Retourne (accuracy, model_name, params_dict)
    """


    y_pred = estimator.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred, average="macro")

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
    elements.append(Paragraph(f"<b>F1 (test)&nbsp;:</b> {f1:.4f}", normal))
    elements.append(Spacer(1, 6))

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

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    doc.build(elements)
    return f1, acc, model_name, params