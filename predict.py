import importlib
import os
import tempfile
import subprocess
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from huggingface_hub import snapshot_download

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        cog_version = importlib.metadata.version("cog")
        print(f"Cog version: {cog_version}\n")

        # Download xcodec_mini_infer
        folder_path = "./inference/xcodec_mini_infer"
        os.makedirs(folder_path, exist_ok=True)
        snapshot_download(repo_id="m-a-p/xcodec_mini_infer", local_dir=folder_path)

    def predict(
        self,
        genre_description: str = Input(
            description="Text containing genre tags that describe the musical style (e.g. instrumental, genre, mood, vocal timbre, vocal gender)",
            default="inspiring female uplifting pop airy vocal electronic bright vocal vocal",
        ),
        lyrics: str = Input(
            description="Lyrics for music generation. Must be structured in segments with [verse], [chorus], etc tags",
            default="",
        ),
        num_segments: int = Input(
            description="Number of segments to generate", default=2, ge=1, le=10
        ),
        max_new_tokens: int = Input(
            description="Maximum number of new tokens to generate",
            default=1500,
            ge=500,
            le=3000,
        ),
    ) -> List[Path]:
        """Run YuE inference on the provided inputs"""

        # Create temporary files for genre and lyrics
        def create_temp_file(content: str, prefix: str) -> str:
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, mode="w", prefix=prefix, suffix=".txt"
            )
            content = content.strip() + "\n\n"
            content = content.replace("\r\n", "\n").replace("\r", "\n")
            temp_file.write(content)
            temp_file.close()
            return temp_file.name

        genre_file = create_temp_file(genre_description, "genre_")
        lyrics_file = create_temp_file(lyrics, "lyrics_")

        # Setup output directory
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)

        # Empty output directory
        for item in os.listdir(output_dir):
            path = os.path.join(output_dir, item)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

        try:
            # Change to inference directory
            os.chdir("./inference")

            # Run inference
            command = [
                "python",
                "infer.py",
                "--stage1_model",
                "m-a-p/YuE-s1-7B-anneal-en-cot",
                "--stage2_model",
                "m-a-p/YuE-s2-1B-general",
                "--genre_txt",
                genre_file,
                "--lyrics_txt",
                lyrics_file,
                "--run_n_segments",
                str(num_segments),
                "--stage2_batch_size",
                "4",
                "--output_dir",
                output_dir,
                "--cuda_idx",
                "0",
                "--max_new_tokens",
                str(max_new_tokens),
            ]

            subprocess.run(command, check=True)

            # Change back to root directory
            os.chdir("..")

            # Find output files
            output_files = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.endswith(".mp3"):
                        output_files.append(Path(os.path.join(root, file)))

            return output_files

        finally:
            # Cleanup temp files
            os.remove(genre_file)
            os.remove(lyrics_file)

    def seed_or_random_seed(self, seed: int | None) -> int:
        # Max seed is 2147483647
        if not seed or seed <= 0:
            seed = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF

        print(f"Using seed: {seed}\n")
        return seed
