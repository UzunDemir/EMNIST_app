from transformers import TrOCRProcessor, VisionEncoderDecoderModel

tokenizer = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
