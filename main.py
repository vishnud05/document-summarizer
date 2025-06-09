from summarizers import BasicPromptSummarizer, TemplateDrivenSummarizer
from dotenv import load_dotenv
load_dotenv()

basic_summarizer = BasicPromptSummarizer(
    model_name="gemma2-9b-it"
)
template_summarizer = TemplateDrivenSummarizer(
    model_name="gemma2-9b-it"
)


with open("doc.txt", "r") as file:
    document = file.read()
    
basic_summary = basic_summarizer(document)
print("Basic Summary:", basic_summary["summary"])
# print("\n" + "="*50 + "\n")
# print("Metadata:", basic_summary["metadata"])
print("\n" + "="*50 + "\n")
template_summary = template_summarizer(document)
print("Template Summary:", template_summary["summary"])
# print("\n" + "="*50 + "\n")
# print("Metadata:", template_summary["metadata"])