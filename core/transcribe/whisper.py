import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class Whisper:
    """Whisper - audio transcriber class"""

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def __init__(self, model_id: str = "openai/whisper-base") -> None:
        self.model_id = model_id
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)

    @property
    def model_name(self):
        """
        Getter method for retrieving the model name.
        """
        return self.model_id

    def save(self, save_dir: str):
        """
        Saves the model and processor to the specified directory.

        Args:
            save_dir (str): The directory where the model and processor will be saved.
        """
        self.model.save_pretrained(f"{save_dir}/model")
        self.processor.save_pretrained(f"{save_dir}/processor")

    def load(self, load_dir: str):
        """
        Load the model and processor from the specified directory.

        Args:
            load_dir (str): The directory from which to load the model and processor.
        """
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(f"{load_dir}/model")
        self.processor = AutoProcessor.from_pretrained(f"{load_dir}/processor")

        self.model.to(self.device)

    def pipeline(self):
        pipe = pipeline(
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

        return pipe
