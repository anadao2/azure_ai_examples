from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.speech import SpeechConfig, SpeechRecognizer
import openai

class AzureServiceIntegration:
    def __init__(self, config):
        self.search_client = SearchClient(
            endpoint=config['search_endpoint'],
            index_name=config['search_index_name'],
            credential=AzureKeyCredential(config['search_api_key'])
        )
        self.form_recognizer_client = DocumentAnalysisClient(
            endpoint=config['form_recognizer_endpoint'],
            credential=AzureKeyCredential(config['form_recognizer_key'])
        )
        self.vision_client = ComputerVisionClient(
            endpoint=config['vision_endpoint'],
            credentials=AzureKeyCredential(config['vision_api_key'])
        )
        self.speech_config = SpeechConfig(
            subscription=config['speech_api_key'],
            region=config['speech_region']
        )
        openai.api_key = config['openai_key']
        openai.api_base = config['openai_endpoint']

    def search_documents(self, query):
        results = self.search_client.search(search_text=query)
        return [(result['id'], result['content']) for result in results]

    def extract_information_from_invoice(self, invoice_path):
        with open(invoice_path, "rb") as f:
            poller = self.form_recognizer_client.begin_analyze_document("prebuilt-invoice", document=f)
        result = poller.result()
        return {field: result.documents[0].fields[field].content for field in result.documents[0].fields}

    def analyze_image(self, image_path):
        with open(image_path, "rb") as image:
            analysis = self.vision_client.analyze_image_in_stream(image, visual_features=["Categories", "Tags", "Description"])
        return {
            "Categories": analysis.categories,
            "Tags": [tag.name for tag in analysis.tags],
            "Description": analysis.description.captions[0].text
        }

    def generate_marketing_text(self, prompt):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()

    def transcribe_audio(self, audio_path):
        recognizer = SpeechRecognizer(speech_config=self.speech_config)
        result = recognizer.recognize_once_from_file(audio_path)
        return result.text

# Configurações
config = {
    'search_endpoint': "https://<your-search-endpoint>.search.windows.net",
    'search_api_key': "<your-search-api-key>",
    'search_index_name': "your-index-name",
    'form_recognizer_endpoint': "https://<your-form-recognizer-endpoint>.cognitiveservices.azure.com/",
    'form_recognizer_key': "<your-form-recognizer-key>",
    'vision_endpoint': "https://<your-computer-vision-endpoint>.cognitiveservices.azure.com/",
    'vision_api_key': "<your-vision-api-key>",
    'speech_api_key': "<your-speech-api-key>",
    'speech_region': "<your-speech-region>",
    'openai_key': "<your-openai-key>",
    'openai_endpoint': "https://<your-openai-endpoint>.openai.azure.com/"
}

# Uso dos Serviços
azure_services = AzureServiceIntegration(config)

# Exemplo 1: Buscar documentos
documents = azure_services.search_documents("Relatório financeiro 2024")
print(documents)

# Exemplo 2: Extrair informações de uma fatura
invoice_data = azure_services.extract_information_from_invoice("path/to/your/invoice.pdf")
print(invoice_data)

# Exemplo 3: Analisar uma imagem
image_analysis = azure_services.analyze_image("path/to/your/image.jpg")
print(image_analysis)

# Exemplo 4: Gerar texto de marketing
marketing_text = azure_services.generate_marketing_text("Write a promotional text for a new eco-friendly water bottle")
print(marketing_text)

# Exemplo 5: Transcrever áudio
transcribed_text = azure_services.transcribe_audio("path/to/your/audio.wav")
print(transcribed_text)
