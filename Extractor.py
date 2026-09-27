import glob
from pathlib import Path
from typing import Generator, Any, List

from utils import SourceCodeFileData


class Extractor:
    def __init__(self, base_directory_path: str):
        self.base_dir_path = base_directory_path
        self.generator = self._extract_next()

    def reset_generator(self):
        self.generator = self._extract_next()

    def _extract_next(self) -> Generator[SourceCodeFileData, Any, None]:
        files = glob.glob(f'{self.base_dir_path}/**/*.java', recursive=True)
        for file in files:
            if _is_implementation_source(file):
                print(f"Extracted file: {file}")
                with open(file, 'r', encoding='utf-8') as java_file:
                    yield SourceCodeFileData(
                        file_path=file,
                        file_content='\n'.join(java_file.readlines())
                    )

    def extract(self) -> SourceCodeFileData | None:
        try:
            return next(self.generator)
        except StopIteration:
            print('All files extracted')
            return None

    def extract_all(self) -> List[SourceCodeFileData]:
        res = []
        file_data = self.extract()
        while file_data is not None:
            res.append(file_data)
            file_data = self.extract()
        print(f'Extracted {len(res)} files')
        return res


def _is_implementation_source(file_path: str) -> bool:
    parts = Path(file_path).parts
    return "src" in parts and "test" not in parts
