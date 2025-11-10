"""
Back Translation (역번역) 모듈

텍스트를 역번역하여 데이터 증강을 수행합니다.
한국어 → 일본어 → 한국어 순서로 번역하여 원본과 유사한 의미의 새로운 텍스트를 생성합니다.

사용 예시:
    from data_processing.back_translation import back_translate_data
    
    augmented_data = back_translate_data(
        input_csv_path="dataset/train.csv",
        output_csv_path="dataset/train_augmented.csv",
        api_key="your-api-key"
    )
"""

import pandas as pd
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from pathlib import Path


class BackTranslator:
    """
    역번역 수행 클래스
    
    텍스트를 한국어 → 일본어 → 한국어로 왕복 번역하여 데이터 증강을 수행합니다.
    """
    
    def __init__(self, api_key: str, max_workers: int = 5):
        """
        Args:
            api_key: OpenAI API 키
            max_workers: 동시 번역 작업 수 (기본값: 5)
        """
        self.client = OpenAI(api_key=api_key)
        self.max_workers = max_workers
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    def translate_round_trip(self, text: str) -> tuple[str | None, str | None]:
        """
        왕복 번역: 한국어 → 일본어 → 한국어
        
        Args:
            text: 원본 텍스트 (한국어)
            
        Returns:
            (일본어_번역본, 역번역된_한국어_텍스트) 튜플
        """
        if not isinstance(text, str) or not text:
            return None, None

        try:
            # 1단계: 한국어 → 일본어
            japanese_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a translator. Translate the given Korean text into Japanese. Always provide the translation without any other explanations or refusals."},
                    {"role": "user", "content": text}
                ],
                timeout=60
            )
            japanese_text = japanese_response.choices[0].message.content.strip()

            # 2단계: 일본어 → 한국어
            korean_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a translator. Translate the given Japanese text into Korean. Always provide the translation without any other explanations or refusals."},
                    {"role": "user", "content": japanese_text}
                ],
                timeout=60
            )
            retranslated_text = korean_response.choices[0].message.content.strip()

            return japanese_text, retranslated_text

        except Exception as e:
            print(f"⚠️ 번역 오류 (텍스트: '{text[:20]}...'): {e}")
            return None, None
    
    def translate_batch(self, texts: list[str]) -> tuple[list[str | None], list[str | None]]:
        """
        여러 텍스트를 병렬로 역번역
        
        Args:
            texts: 번역할 텍스트 리스트
            
        Returns:
            (일본어_번역본_리스트, 역번역된_한국어_리스트) 튜플
        """
        japanese_results = [None] * len(texts)
        korean_results = [None] * len(texts)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self.translate_round_trip, text): i
                for i, text in enumerate(texts)
            }

            for future in tqdm(as_completed(future_to_index), total=len(texts), desc="역번역 진행 중"):
                idx = future_to_index[future]
                try:
                    japanese_text, korean_text = future.result()
                    japanese_results[idx] = japanese_text
                    korean_results[idx] = korean_text
                except Exception as e:
                    print(f"⚠️ 인덱스 {idx} 번역 실패: {e}")

        return japanese_results, korean_results
    
    def calculate_similarity(self, original_texts: list[str], retranslated_texts: list[str]) -> list[float]:
        """
        원본과 역번역 텍스트 간 코사인 유사도 계산
        
        Args:
            original_texts: 원본 텍스트 리스트
            retranslated_texts: 역번역된 텍스트 리스트
            
        Returns:
            코사인 유사도 리스트 (0.0 ~ 1.0)
        """
        # None을 빈 문자열로 변환
        originals = [text if text else '' for text in original_texts]
        retranslated = [text if text else '' for text in retranslated_texts]

        # 임베딩 생성
        emb_original = self.model.encode(originals, convert_to_tensor=True)
        emb_retranslated = self.model.encode(retranslated, convert_to_tensor=True)

        # 코사인 유사도 계산
        similarities = util.cos_sim(emb_original, emb_retranslated)
        
        # 대각선 값 추출 (각 텍스트 쌍의 유사도)
        return [similarities[i][i].item() for i in range(len(original_texts))]


class LanguageDetector:
    """
    언어 감지 유틸리티 클래스
    
    텍스트에 포함된 특정 언어의 고유 문자를 검사하여 언어를 감지합니다.
    """
    
    @staticmethod
    def is_french(text: str) -> bool:
        """프랑스어 감지 (à, â, ç, è, é, ê, î, ô, ù, û, œ 등)"""
        if not isinstance(text, str):
            return False
        return bool(re.search(
            r'[\u00e0\u00e2\u00e7\u00e8\u00e9\u00ea\u00ee\u00f4\u0153\u00f9\u00fb'  # 소문자
            r'\u00c0\u00c2\u00c7\u00c8\u00c9\u00ca\u00ce\u00d4\u0152\u00d9\u00db]', # 대문자
            text
        ))
    
    @staticmethod
    def is_korean(text: str) -> bool:
        """한국어 감지 (한글)"""
        if not isinstance(text, str):
            return False
        return bool(re.search(r'[\uac00-\ud7af]', text))
    
    @staticmethod
    def is_japanese(text: str) -> bool:
        """일본어 감지 (히라가나, 가타카나, 한자)"""
        if not isinstance(text, str):
            return False
        return bool(re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', text))
    
    @staticmethod
    def is_chinese(text: str) -> bool:
        """중국어 감지 (한자)"""
        if not isinstance(text, str):
            return False
        return bool(re.search(r'[\u4e00-\u9fff]', text))
    
    @staticmethod
    def is_english(text: str) -> bool:
        """영어 감지"""
        if not isinstance(text, str):
            return False
        return bool(re.search(r'[a-zA-Z]', text))
    
    @staticmethod
    def is_german(text: str) -> bool:
        """독일어 감지 (ä, ö, ü, ß 등)"""
        if not isinstance(text, str):
            return False
        return bool(re.search(r'[\u00e4\u00f6\u00fc\u00c4\u00d6\u00dc\u00df]', text))


def back_translate_data(
    input_csv_path: str,
    output_csv_path: str,
    api_key: str,
    label_value: int = 1,
    similarity_threshold: float = 0.7,
    max_similarity: float = 0.96,
    max_workers: int = 5
) -> pd.DataFrame:
    """
    역번역을 통한 데이터 증강
    
    지정된 라벨의 텍스트를 역번역하여 원본 데이터에 추가합니다.
    유사도가 낮거나 일본어가 포함된 경우 자동으로 재번역합니다.
    
    Args:
        input_csv_path: 입력 CSV 파일 경로 (상대경로)
        output_csv_path: 출력 CSV 파일 경로 (상대경로)
        api_key: OpenAI API 키
        label_value: 역번역할 라벨 값 (기본값: 1)
        similarity_threshold: 최소 유사도 (기본값: 0.7)
        max_similarity: 최대 유사도 (기본값: 0.96)
        max_workers: 동시 작업 수 (기본값: 5)
        
    Returns:
        증강된 데이터프레임
    """
    base_path = Path(__file__).parent.parent
    input_path = base_path / input_csv_path
    output_path = base_path / output_csv_path
    
    # 데이터 로드
    print(f"📂 데이터 로딩: {input_path}")
    data = pd.read_csv(input_path)
    
    # 지정된 라벨의 데이터만 추출
    filtered = data[data["label"] == label_value].copy()
    print(f"📊 역번역 대상: {len(filtered)}개 (라벨={label_value})")
    
    # 역번역기 초기화
    translator = BackTranslator(api_key=api_key, max_workers=max_workers)
    detector = LanguageDetector()
    
    # 1차 역번역
    print("🔄 1차 역번역 수행 중...")
    texts = filtered["text"].tolist()
    japanese_results, korean_results = translator.translate_batch(texts)
    
    filtered['dialogue_translated'] = japanese_results
    filtered['dialogue_retranslated'] = korean_results
    
    # 유사도 계산
    print("📐 유사도 계산 중...")
    similarities = translator.calculate_similarity(
        filtered["text"].tolist(),
        filtered["dialogue_retranslated"].tolist()
    )
    filtered["cosine_similarity"] = similarities
    
    # 재번역 필요 여부 확인
    has_japanese = filtered['dialogue_retranslated'].apply(detector.is_japanese)
    low_similarity = filtered['cosine_similarity'] < similarity_threshold
    needs_retry = has_japanese | low_similarity
    
    correct_df = filtered[~needs_retry]
    retry_df = filtered[needs_retry].copy()
    
    print(f"📊 재번역 필요: {len(retry_df)}개 (일본어 {has_japanese.sum()}건, 유사도 미달 {low_similarity.sum()}건)")
    
    # 재번역 수행
    if not retry_df.empty:
        print("🔄 재번역 수행 중...")
        retry_texts = retry_df["text"].tolist()
        retry_japanese, retry_korean = translator.translate_batch(retry_texts)
        
        retry_df['dialogue_translated'] = retry_japanese
        retry_df['dialogue_retranslated'] = retry_korean
        
        retry_similarities = translator.calculate_similarity(
            retry_df["text"].tolist(),
            retry_df["dialogue_retranslated"].tolist()
        )
        retry_df["cosine_similarity"] = retry_similarities
        
        final_df = pd.concat([correct_df, retry_df], ignore_index=True)
        print("✅ 재번역 완료")
    else:
        final_df = filtered.copy()
        print("✅ 모든 데이터가 조건을 만족합니다")
    
    # 유사도 필터링 (임계값 범위 내만 선택)
    final_df = final_df[
        (final_df.cosine_similarity > similarity_threshold) & 
        (final_df.cosine_similarity < max_similarity)
    ]
    
    # 결과 정리
    final_df = final_df[['dialogue_retranslated', 'label']].copy()
    final_df.columns = ['text', 'label']
    
    # 원본과 병합
    augmented_df = pd.concat([data, final_df], ignore_index=True)
    
    print(f"💾 결과 저장: {output_path}")
    print(f"📊 최종 데이터: {augmented_df.shape[0]}개 (원본 {len(data)}개 + 증강 {len(final_df)}개)")
    augmented_df.to_csv(output_path, index=False)
    
    return augmented_df


if __name__ == "__main__":
    import os
    
    # API 키 설정 (환경변수 또는 직접 입력)
    api_key = os.getenv("OPENAI_API_KEY", "")
    
    if not api_key:
        print("⚠️ 경고: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("환경변수를 설정하거나 api_key를 직접 입력해주세요.")
        exit(1)
    
    # 역번역 수행
    augmented_data = back_translate_data(
        input_csv_path="dataset/stt_results_train.csv",
        output_csv_path="dataset/spam_stt_bt_fr.csv",
        api_key=api_key,
        label_value=1,
        similarity_threshold=0.7,
        max_similarity=0.96,
        max_workers=5
    )
    
    print("\n✅ 역번역 작업 완료!")

