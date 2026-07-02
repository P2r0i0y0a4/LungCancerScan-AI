import os
import smtplib
import tempfile
from datetime import datetime
from email.message import EmailMessage
from dotenv import load_dotenv

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

load_dotenv()
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
DOCTOR_EMAIL = os.getenv("DOCTOR_EMAIL")

def draw_border(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#0f172a"))
    canvas.setLineWidth(2)
    canvas.rect(30, 30, A4[0]-60, A4[1]-60)
    canvas.setLineWidth(0.5)
    canvas.rect(35, 35, A4[0]-70, A4[1]-70)
    canvas.restoreState()

def generate_pdf(p_name, p_id, p_age, p_gender, diag, risk_score, exec_summary, reco, img):
    pdf_fd, pdf_path = tempfile.mkstemp(suffix=".pdf")
    os.close(pdf_fd)
    
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    
    # --- UI STYLES (STRICTLY UNCHANGED) ---
    title_style = ParagraphStyle('Title', fontSize=22, textColor=colors.HexColor("#0284c7"), alignment=1, spaceAfter=5, fontName="Helvetica-Bold")
    subtitle_style = ParagraphStyle('Sub', fontSize=10, textColor=colors.grey, alignment=1, spaceAfter=20)
    section_h = ParagraphStyle('SectionH', fontSize=12, textColor=colors.white, backColor=colors.HexColor("#1e293b"), leftIndent=5, spaceBefore=10, spaceAfter=10, fontName="Helvetica-Bold", leading=16)
    summary_box_style = ParagraphStyle('Summary', fontSize=11, leading=16, textColor=colors.HexColor("#334155"), alignment=0)
    label_style = ParagraphStyle('Label', fontSize=10, fontName="Helvetica-Bold", textColor=colors.HexColor("#64748b"))

    elements = []
    elements.append(Paragraph("LUNGSCAN AI | CLINICAL REPORT", title_style))
    elements.append(Paragraph("AI-Assisted Multimodal Diagnostic Support System", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#38bdf8"), spaceAfter=15))

    pt_data = [
        [Paragraph("PATIENT NAME", label_style), p_name.upper(), Paragraph("PATIENT ID", label_style), f"ID-{p_id}"],
        [Paragraph("AGE", label_style), str(p_age), Paragraph("GENDER", label_style), p_gender.upper()],
        [Paragraph("GEN. DATE", label_style), datetime.now().strftime('%Y-%m-%d'), Paragraph("GEN. TIME", label_style), datetime.now().strftime('%H:%M')]
    ]
    pt_table = Table(pt_data, colWidths=[1.2*inch, 1.9*inch, 1.2*inch, 1.9*inch])
    pt_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('PADDING', (0,0), (-1,-1), 7)]))
    elements.append(pt_table)

    # --- LOGIC FIX: Diagnostic Support ---
    elements.append(Paragraph(" 📋 DIAGNOSTIC DECISION SUPPORT", section_h))
    res_color = colors.HexColor("#dc2626") if diag.lower() == "malignant" else colors.HexColor("#16a34a")
    
    res_data = [
        [Paragraph("FINAL AI PATHOLOGY CONCLUSION:", label_style), Paragraph(f"<b>{diag.upper()}</b>", ParagraphStyle('Res', fontSize=14, textColor=res_color))],
        [Paragraph("RISK ASSESSMENT LEVEL:", label_style), risk_score]
    ]
    res_table = Table(res_data, colWidths=[2.5*inch, 3.7*inch])
    res_table.setStyle(TableStyle([('BOTTOMPADDING', (0,0), (-1,-1), 10), ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#f1f5f9"))]))
    elements.append(res_table)

    # --- LOGIC FIX: Summary (Using pre-computed summaries from app.py) ---
    elements.append(Paragraph(" 🩺 CLINICAL EXECUTIVE SUMMARY", section_h))
    summary_content = f"{exec_summary}<br/><br/><b>AI Clinical Advice:</b> {reco}"
    elements.append(Paragraph(summary_content, summary_box_style))

    # --- RADIOLOGY IMAGE ---
    elements.append(Spacer(1, 15))
    img_fd, img_path = tempfile.mkstemp(suffix=".png")
    os.close(img_fd)
    img.save(img_path)
    im = RLImage(img_path, width=2.5*inch, height=2.5*inch)
    img_table = Table([[im]])
    img_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#0f172a"))]))
    elements.append(img_table)
    elements.append(Paragraph("<alignment=center><i>Figure 1: Analyzed Pulmonary CT Morphology</i></alignment>", styles['Italic']))

    # --- FOOTER ---
    elements.append(Spacer(1, 40))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    disclaimer = "<b>Disclaimer:</b> This AI-generated report is for clinical decision support. Final diagnosis must be performed by a board-certified medical professional."
    elements.append(Paragraph(disclaimer, ParagraphStyle('Disc', fontSize=8, textColor=colors.grey, alignment=1)))

    doc.build(elements, onFirstPage=draw_border)
    if os.path.exists(img_path): os.remove(img_path)
    return pdf_path

def send_email(patient_email, patient_name, pdf_path):
    msg = EmailMessage()
    msg["Subject"] = f"CONFIDENTIAL: LungScan AI Diagnostic Report - {patient_name}"
    msg["From"] = EMAIL_SENDER
    # Sends to both Doctor and Patient
    msg["To"] = f"{patient_email}, {DOCTOR_EMAIL}"
    msg.set_content(f"Please find the attached LungScan AI diagnostic report for {patient_name}.")
    
    with open(pdf_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename=f"LungScan_Report_{patient_name}.pdf")

    # This specific structure clears the (334) error
    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.ehlo()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        raise Exception(f"Mail Server Error: {str(e)}")