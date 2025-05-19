import os
import shutil
import imaplib
import email
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

# Load environment variables
load_dotenv()

# Paths
CLOUD_FOLDER = "cloud_folder/proposals/"
DRAFT_FOLDER = "cloud_folder/drafts/"
APPROVED_FOLDER = "cloud_folder/approved/"

# LangChain LLM setup
llm = ChatOpenAI(temperature=0.1, model_name="gpt-4")
embeddings = OpenAIEmbeddings()

# Load and vectorize past proposals
def load_proposals():
    loader = DirectoryLoader(CLOUD_FOLDER, glob="**/*.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    vectordb = FAISS.from_documents(docs, embeddings)
    return vectordb

# Extract client interest areas from email
def extract_services(email_text):
    prompt = f"Extract Zoho services of interest from this email: {email_text}. Return as a comma-separated list."
    return llm.predict(prompt)

# Generate a proposal in existing structure style
def generate_proposal(services, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    
    prompt = f"Using the format and tone of previous proposals, write a detailed proposal for these Zoho services: {services}. Include sections like Introduction, Service Details, Plan Recommendations, Pricing, Next Steps, and Contact Info."
    
    response = qa_chain.run(prompt)
    return response

# Save proposal to draft folder
def save_proposal(proposal_text, filename):
    with open(os.path.join(DRAFT_FOLDER, filename), "w") as f:
        f.write(proposal_text)

# Move approved proposal to approved folder
def approve_proposal(filename):
    src = os.path.join(DRAFT_FOLDER, filename)
    dest = os.path.join(APPROVED_FOLDER, filename)
    shutil.move(src, dest)

# Send final email to client
def send_email(to_email, proposal_text):
    msg = MIMEMultipart()
    msg['From'] = os.getenv("EMAIL_ADDRESS")
    msg['To'] = to_email
    msg['Subject'] = "Your Zoho Services Proposal"
    msg.attach(MIMEText(proposal_text, 'plain'))

    server = smtplib.SMTP(os.getenv("SMTP_SERVER"), 587)
    server.starttls()
    server.login(os.getenv("EMAIL_ADDRESS"), os.getenv("EMAIL_PASSWORD"))
    server.send_message(msg)
    server.quit()

# Example execution
if __name__ == "__main__":
    vectordb = load_proposals()

    # Simulating incoming email content
    email_text = "Hi team, I'm interested in Zoho CRM and Zoho Books for my business. Please send a quote."

    services = extract_services(email_text)
    proposal = generate_proposal(services, vectordb)
    save_proposal(proposal, "proposal_client1.txt")

    # Sales team approves manually -> move file
    approve_proposal("proposal_client1.txt")

    # Send email to client
    with open(os.path.join(APPROVED_FOLDER, "proposal_client1.txt"), "r") as f:
        final_proposal = f.read()

    send_email("client@example.com", final_proposal)
