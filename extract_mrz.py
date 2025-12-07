#!/usr/bin/env python3

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List
import re


class MRZExtractor:

    def __init__(self):
        self.tesseract = None
        self.easyocr = None

        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.tesseract = pytesseract
        except:
            pass

        try:
            import easyocr
            self.easyocr = easyocr.Reader(['en'], gpu=False, verbose=False)
        except:
            pass

        if not self.tesseract and not self.easyocr:
            raise RuntimeError("Install pytesseract or easyocr")

    def extract(self, image_path: str) -> Tuple[str, str]:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read: {image_path}")

        h, w = image.shape[:2]
        mrz_region = image[int(h * 0.7):, :]

        all_lines = []

        for img in self._preprocess(mrz_region):
            all_lines.extend(self._ocr(img))

        for img in self._preprocess(image):
            all_lines.extend(self._ocr(img))

        line1, line2 = self._select_mrz_lines(all_lines)

        return line1, line2

    def _preprocess(self, image: np.ndarray) -> List[np.ndarray]:
        """Generate preprocessed variants."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        h, w = gray.shape[:2]
        if w < 800:
            scale = 800 / w
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        variants = [gray]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        variants.append(enhanced)

        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)

        adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 31, 10)
        variants.append(adaptive)

        return variants

    def _ocr(self, image: np.ndarray) -> List[str]:
        lines = []

        if self.tesseract:
            try:
                config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
                text = self.tesseract.image_to_string(image, config=config)
                for line in text.split('\n'):
                    cleaned = self._clean(line)
                    if len(cleaned) >= 30:
                        lines.append(cleaned)
            except:
                pass

        if self.easyocr:
            try:
                results = self.easyocr.readtext(image)
                for (_, text, _) in results:
                    cleaned = self._clean(text)
                    if len(cleaned) >= 30:
                        lines.append(cleaned)
            except:
                pass

        return lines

    def _clean(self, text: str) -> str:
        text = text.upper().strip()

        replacements = {
            ' ': '', '\n': '', '\t': '',
            '«': '<', '»': '<', '‹': '<', '›': '<',
            '(': '<', ')': '<', '[': '<', ']': '<',
            '{': '<', '}': '<', '-': '<', '_': '<',
            '.': '', ',': '', ':': '', ';': '',
            '|': 'I', '!': 'I',
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        valid = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<')
        text = ''.join(c for c in text if c in valid)

        text = re.sub(r'[KRX]{3,}$', lambda m: '<' * len(m.group()), text)
        text = re.sub(r'<[KRX]+<', lambda m: '<' * len(m.group()), text)
        text = re.sub(r'<[KRX]{2,}$', lambda m: '<' * len(m.group()), text)

        return text

    def _fix_nationality_codes(self, line2: str) -> str:
        if len(line2) < 13:
            return line2

        nationality = line2[10:13]
        fixed_nat = ''
        for c in nationality:
            if c == '0':
                fixed_nat += 'O'
            else:
                fixed_nat += c

        return line2[:10] + fixed_nat + line2[13:]

    def _select_mrz_lines(self, lines: List[str]) -> Tuple[str, str]:
        """Select best line1 and line2."""
        line1_candidates = []
        line2_candidates = []

        for line in lines:
            if not line or len(line) < 30:
                continue

            if line[0] in 'PIV' and '<' in line[:10]:
                line1_candidates.append(line)
            elif line[0].isalnum() and sum(c.isdigit() for c in line[:10]) >= 3:
                line2_candidates.append(line)

        line1 = self._vote(line1_candidates)
        line2 = self._vote(line2_candidates)

        line1 = (line1 + '<' * 44)[:44]
        line2 = (line2 + '<' * 44)[:44]

        line2 = self._fix_nationality_codes(line2)

        line1 = self._fix_trailing_fillers(line1)

        return line1, line2

    def _fix_trailing_fillers(self, line: str) -> str:
        match = re.search(r'<<[A-Z]+<<', line)
        if match:
            end_pos = match.end()
            rest = line[end_pos:]
            if rest.count('<') > len(rest) * 0.7:
                fixed_rest = re.sub(r'[A-Z0-9]', '<', rest)
                line = line[:end_pos] + fixed_rest

        return line

    def _vote(self, candidates: List[str]) -> str:
        if not candidates:
            return ''
        if len(candidates) == 1:
            return candidates[0]

        max_len = max(len(c) for c in candidates)
        result = []

        for i in range(max_len):
            chars = [c[i] for c in candidates if i < len(c)]
            if chars:
                from collections import Counter
                result.append(Counter(chars).most_common(1)[0][0])

        return ''.join(result)


def main():
    import sys
    import json

    if len(sys.argv) < 2:
        #print("Usage: python extract_mrz.py <image1> [image2] ...")
        #print("       python extract_mrz.py --json <image1> [image2] ...")
        sys.exit(1)

    json_output = '--json' in sys.argv
    images = [a for a in sys.argv[1:] if not a.startswith('--')]

    extractor = MRZExtractor()
    results = []

    for image_path in images:
        try:
            line1, line2 = extractor.extract(image_path)
            results.append({
                'file': Path(image_path).name,
                'mrz': [line1, line2]
            })

            if not json_output:
                print(f"\n{Path(image_path).name}")
                print("-" * 44)
                print(line1)
                print(line2)
        except Exception as e:
            results.append({
                'file': Path(image_path).name,
                'error': str(e)
            })
            if not json_output:
                print(f"\n{Path(image_path).name}: Error - {e}")

    if json_output:
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
