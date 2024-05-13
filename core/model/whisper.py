from core.model.base import Model


class Whisper(Model):
    """Whisper - audio transcriber class"""

    device: str = "cpu"
    torch_dtype = None

    def __init__(self, model_id: str = "openai/whisper-base") -> None:
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except ImportError:
            raise ImportError(
                "Failed to import transformers. Install using `pip install transformers`."
            )
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model_id = model_id
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=15,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    @property
    def model_name(self):
        """
        Getter method for retrieving the model name.
        """
        return self.model_id

    def run(self, audio_file: bytes):
        return self.pipeline(audio_file)
