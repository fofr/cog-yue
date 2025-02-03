import importlib
import os
import tempfile
import subprocess
import shutil
import time
from typing import List
from cog import BasePredictor, Input, Path

WEIGHTS_BASE_URL = "https://weights.replicate.delivery/default/yue/"


class Predictor(BasePredictor):
    def download_weights(self, filename: str, dest_dir: str):
        os.makedirs(dest_dir, exist_ok=True)

        if not os.path.exists(f"{dest_dir}/{filename}"):
            print(f"⏳ Downloading {filename} to {dest_dir}")
            start = time.time()
            subprocess.check_call(
                [
                    "pget",
                    "--log-level",
                    "warn",
                    "-xf",
                    f"{WEIGHTS_BASE_URL}/{filename}.tar",
                    dest_dir,
                ],
                close_fds=False,
            )
            print(f"✅ Download completed in {time.time() - start:.2f} seconds")
        else:
            print(f"✅ {filename} already exists in {dest_dir}")

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        cog_version = importlib.metadata.version("cog")
        print(f"Cog version: {cog_version}\n")

        self.download_weights(
            "models--m-a-p--YuE-s1-7B-anneal-en-cot", "/src/inference/models"
        )
        self.download_weights("xcodec_mini_infer", "/src/inference")

    def predict(
        self,
        genre_description: str = Input(
            description="Text containing genre tags that describe the musical style (e.g. instrumental, genre, mood, vocal timbre, vocal gender)",
            default="inspiring female uplifting pop airy vocal electronic bright vocal vocal",
        ),
        lyrics: str = Input(
            description="Lyrics for music generation. Must be structured in segments with [verse], [chorus], etc tags",
            default="[verse]\nOh yeah, oh yeah, oh yeah\n\n[chorus]\nOh yeah, oh yeah, oh yeah",
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
        seed: int = Input(
            description="Set a seed for reproducibility. Random by default.",
            default=None,
        ),
    ) -> List[Path]:
        """Run YuE inference on the provided inputs"""
        seed = self.seed_or_random_seed(seed)

        # Validate inputs
        if not lyrics.strip():
            raise ValueError("Lyrics cannot be empty")

        if not any(tag in lyrics.lower() for tag in ["[verse]", "[chorus]"]):
            raise ValueError("Lyrics must contain at least one [verse] or [chorus] tag")

        if not genre_description.strip():
            raise ValueError("Genre description cannot be empty")

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
        output_dir = "/src/output"
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
                "--seed",
                str(seed),
            ]

            subprocess.run(command, check=True)

            # Change back to root directory
            os.chdir("..")

            # Find output files in vocoder/mix directory
            mix_dir = os.path.join(output_dir, "vocoder", "mix")
            output_files = []
            if os.path.exists(mix_dir):
                for file in os.listdir(mix_dir):
                    if file.endswith(".mp3"):
                        output_files.append(Path(os.path.join(mix_dir, file)))

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
