원본페이지: "https://github.com/DS4SD/DocLayNet?tab=readme-ov-file"

# DocLayNet: 문서 레이아웃 분석을 위한 대규모 인적 주석 데이터셋

DocLayNet은 다양한 문서 소스에서 가져온 80,863페이지를 포함하는, 사람이 직접 주석을 단(human-annotated) 문서 레이아웃 분할 데이터셋입니다.

## 뉴스

| 날짜             | 내용                                                                                                                                   |
| :--------------- | :------------------------------------------------------------------------------------------------------------------------------------- |
| **2023년 1월 26일** | DocLayNet 데이터셋이 Hugging Face의 [ds4sd/DocLayNet](https://huggingface.co/datasets/ds4sd/DocLayNet)에서 사용 가능합니다.                         |
| **2023년 1월 13일** | ICDAR 2023에서 기업 문서 레이아웃 분할에 대한 경쟁을 주최합니다. 자세한 내용은 [경쟁 웹사이트](https://ds4sd.github.io/icdar23-doclaynet/)를 참조하세요. |

## 개요

DocLayNet은 6개 문서 카테고리에서 가져온 80,863개의 고유 페이지에 대해 11개의 개별 클래스 레이블에 대한 바운딩 박스(bounding-box)를 사용하여 페이지별 레이아웃 분할 ground-truth를 제공합니다. PubLayNet이나 DocBank 같은 관련 연구와 비교하여 다음과 같은 몇 가지 고유한 특징을 제공합니다:

1.  **인적 주석(Human Annotation)**: DocLayNet은 잘 훈련된 전문가들에 의해 수작업으로 주석 처리되었으며, 각 페이지 레이아웃에 대한 인간의 인식과 해석을 통해 레이아웃 분할의 골드 스탠다드를 제공합니다.
2.  **넓은 레이아웃 다양성(Large layout variability)**: DocLayNet은 금융, 과학, 특허, 입찰, 법률 문서 및 매뉴얼 등 다양한 공공 소스에서 가져온 다양하고 복잡한 레이아웃을 포함합니다.
3.  **상세한 레이블 세트(Detailed label set)**: DocLayNet은 레이아웃 특징을 높은 세부 수준으로 구별하기 위해 11개의 클래스 레이블을 정의합니다.
4.  **중복 주석(Redundant annotations)**: DocLayNet 페이지의 일부는 이중 또는 삼중으로 주석 처리되어, 주석 불확실성과 기계 학습 모델로 달성 가능한 예측 정확도의 상한선을 추정할 수 있습니다.
5.  **사전 정의된 훈련/테스트/검증 세트(Pre-defined train- test- and validation-sets)**: DocLayNet은 각 세트에 대해 고정된 세트를 제공하여 클래스 레이블의 비례적 표현을 보장하고 세트 간 고유한 레이아웃 스타일의 유출을 방지합니다.

## 🤗 Hugging Face로 데이터셋 사용하기

DocLayNet 데이터셋은 Hugging Face의 [ds4sd/DocLayNet](https://huggingface.co/datasets/ds4sd/DocLayNet)에서 사용할 수 있습니다.

```python
>>> from datasets import load_dataset

>>> dataset = load_dataset("ds4sd/DocLayNet")

>>> dataset
DatasetDict({
    train: Dataset({
        features: ['image_id', 'image', 'width', 'height', 'doc_category', 'collection', 'doc_name', 'page_no', 'objects'],
        num_rows: 69375
    })
    validation: Dataset({
        features: ['image_id', 'image', 'width', 'height', 'doc_category', 'collection', 'doc_name', 'page_no', 'objects'],
        num_rows: 6489
    })
    test: Dataset({
        features: ['image_id', 'image', 'width', 'height', 'doc_category', 'collection', 'doc_name', 'page_no', 'objects'],
        num_rows: 4999
    })
})
```

## 다운로드

| 데이터셋             | 레코드 수 | 크기(GB) | URL                                                                                                |
| :------------------- | :-------- | :------- | :------------------------------------------------------------------------------------------------- |
| DocLayNet 코어 데이터셋 | 80,863    | 28 GiB   | [Download](https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip) |
| DocLayNet 추가 파일    | 80,863    | 7.5 GiB  | [Download](https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_extra.zip) |

추가로, 주석 전문가 교육에 사용된 라벨링 가이드라인을 [여기](https://raw.githubusercontent.com/DS4SD/DocLayNet/main/assets/DocLayNet_Labeling_Guide_Public.pdf)에서 제공합니다.

## 데이터셋 구조

DocLayNet은 네 가지 유형의 데이터 자산을 제공합니다:

1.  모든 페이지의 PNG 이미지 (정사각형 `1025 x 1025px`로 크기 조정됨)
2.  각 PNG 이미지에 대한 COCO 형식의 바운딩 박스 주석
3.  추가: 각 PNG 이미지에 해당하는 단일 페이지 PDF 파일
4.  추가: 각 PDF 페이지에 해당하는 JSON 파일 (좌표와 내용이 있는 디지털 텍스트 셀 제공)

데이터셋은 다음 디렉토리 구조로 구성됩니다:

*Doclaynet 코어 데이터셋*

```
├── COCO
│   ├── test.json
│   ├── train.json
│   └── val.json
├── PNG
│   ├── <hash>.png
│   ├── ...
```

*Doclaynet 추가 파일*

```
├── PDF
│   ├── <hash>.pdf
│   ├── ...
├── JSON
│   ├── <hash>.json
│   ├── ...
```

## 데이터 포맷 상세 정보

### 페이지 (Pages)

문서 페이지는 비트맵 이미지(PNG)와 원본 PDF 형식으로 제공됩니다. COCO 형식의 레이아웃 주석은 PNG 이미지만 참조합니다.

### COCO 주석 (Annotations)

바운딩 박스 주석의 경우, DocLayNet은 [여기](https://cocodataset.org/#format-data)에 정의된 표준 COCO 형식 주석을 제공합니다.
각 COCO 이미지 레코드에는 데이터 하위 선택을 허용하고 출처를 제공하기 위한 추가 사용자 정의 필드가 포함되어 있습니다.

#### 예시 페이지 (바운딩 박스 주석 포함)

[![example_page](/DS4SD/DocLayNet/raw/main/assets/132a855ee8b23533d8ae69af0049c038171a06ddfcac892c3c6d7e6b4091c642.png)](/DS4SD/DocLayNet/blob/main/assets/132a855ee8b23533d8ae69af0049c038171a06ddfcac892c3c6d7e6b4091c642.png)

#### 예시 COCO 이미지 레코드

```json
    ...
    {
      "id": 1,
      "width": 1025,
      "height": 1025,
      "file_name": "132a855ee8b23533d8ae69af0049c038171a06ddfcac892c3c6d7e6b4091c642.png",

      // 사용자 정의 필드:
      "doc_category": "financial_reports", // 상위 수준 문서 카테고리
      "collection": "ann_reports_00_04_fancy", // 하위 컬렉션 이름
      "doc_name": "NASDAQ_FFIN_2002.pdf", // 원본 문서 파일명
      "page_no": 9, // 원본 문서 내 페이지 번호
      "precedence": 0, // 주석 순서, 중복 이중 또는 삼중 주석의 경우 0이 아님
    },
    ...
```

`doc_category` 필드는 다음 상수 중 하나를 사용합니다:

```
financial_reports,
scientific_articles,
laws_and_regulations,
government_tenders,
manuals,
patents
```

#### 예시 COCO 주석 레코드

아래에 표시된 주석 레코드는 모두 위에 표시된 페이지 이미지에 속합니다. 모든 바운딩 박스는 별도의 레코드이며, 공통 `image_id`에 매칭됩니다.

<details>
<summary>클릭하여 펼치기...</summary>

```json
  "annotations": [
    {
      "id": 8,
      "image_id": 1,
      "category_id": 1, // Caption
      "bbox": [ 210.06, 31.14, 173.98, 39.27 ], // x,y,w,h
      "segmentation": [ [ 210.06, 31.14, 210.06, 70.41, 384.04, 70.41, 384.04, 31.14 ] ], // polygon [[x1,y1, x2,y2,...]]
      "area": 6832.55,
      "iscrowd": 0,
      "precedence": 0
    },
    {
      "id": 9,
      "image_id": 1,
      "category_id": 7, // Picture
      "bbox": [ 434.93, -0.49, 589.85, 590.23 ],
      "segmentation": [ [ 434.93, -0.49, 434.93, 589.74, 1024.79, 589.74, 1024.79, -0.49 ] ],
      "area": 348154.46,
      "iscrowd": 0,
      "precedence": 0
    },
    {
      "id": 10,
      "image_id": 1,
      "category_id": 8, // Section-header
      "bbox": [ 66.99, 112.10, 290.86, 13.66 ],
      "segmentation": [ [ 66.99, 112.10, 66.99, 125.76, 357.86, 125.76, 357.86, 112.10 ] ],
      "area": 3974.08,
      "iscrowd": 0,
      "precedence": 0
    },
    {
      "id": 11,
      "image_id": 1,
      "category_id": 10, // Text
      "bbox": [ 66.99, 133.58, 325.35, 131.31 ],
      "segmentation": [ [ 66.99, 133.58, 66.99, 264.89, 392.34, 264.89, 392.34, 133.58 ] ],
      "area": 42722.71,
      "iscrowd": 0,
      "precedence": 0
    },
    // ... (다른 주석 레코드들)
    {
      "id": 15,
      "image_id": 1,
      "category_id": 5, // Page-footer
      "bbox": [ 18.32, 1005.24, 7.44, 10.35 ],
      "segmentation": [ [ 18.32, 1005.24, 18.32, 1015.59, 25.77, 1015.59, 25.77, 1005.24 ] ],
      "area": 77.13,
      "iscrowd": 0,
      "precedence": 0
    },
    ...
  ]
```

</details>

### 추가 파일: JSON 파일

DocLayNet은 각 PDF 셀의 바운딩 박스 좌표와 텍스트, 그리고 추가 메타데이터를 포함하는 보조 JSON 파일을 제공합니다. 이 데이터는 PDF에서만 생성되며 주석과는 독립적입니다.

#### 예시 JSON 데이터

아래 스니펫은 위에 표시된 페이지에 대한 JSON 데이터의 일부를 보여줍니다. 표시된 텍스트 셀은 섹션 헤더(인덱스: 3)입니다.

```json
{
  "metadata": {
    "page_hash": "132a855ee8b23533d8ae69af0049c038171a06ddfcac892c3c6d7e6b4091c642", // 고유 식별자, 파일명과 동일
    "original_filename": "NASDAQ_FFIN_2002.pdf", // 원본 문서 파일명
    "page_no": 9, // 원본 문서 내 페이지 번호
    "num_pages": 28, // 원본 문서의 총 페이지 수
    "original_width": 612, // 72ppi 기준 픽셀 너비
    "original_height": 792, // 72ppi 기준 픽셀 높이
    "coco_width": 1025, // PNG 및 COCO 형식의 픽셀 너비
    "coco_height": 1025, // PNG 및 COCO 형식의 픽셀 높이
    "collection": "ann_reports_00_04_fancy", // 하위 컬렉션 이름
    "doc_category": "financial_reports" // 상위 수준 문서 카테고리
  },
  "cells": [ // 디지털 PDF 데이터의 모든 텍스트 셀
    {
      // 텍스트 셀의 바운딩 박스 좌표,
      // [x,y,w,h] 형식 (COCO 주석과 동일)
      // 여기서 (x,y)는 왼쪽 상단 모서리,
      //       (w,h)는 너비와 높이
      // (0,0, coco_width, coco_height) 좌표 공간 기준
      "bbox": [
        66.99346405228758,
        112.10344760101009,
        290.869358251634,
        13.66279703282828
      ],
      "text": "Leigh Taliaferro, M.D., values consistency.", // 셀의 문자열 내용
      "font": {
        "color": [ 12, 72, 142, 255 ], // RGBA 색상
        "name": "/AAAAAC+HelveticaNeue-Medium", // 폰트 이름
        "size": 1 // 폰트 크기 (상대적일 수 있음)
      }
    },
    ...
  ]
}
```

## 논문 (Paper)

**"DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis"** (KDD 2022).

*   Birgit Pfitzmann ([bpf@zurich.ibm.com](mailto:bpf@zurich.ibm.com))
*   Christoph Auer ([cau@zurich.ibm.com](mailto:cau@zurich.ibm.com))
*   Michele Dolfi ([dol@zurich.ibm.com](mailto:dol@zurich.ibm.com))
*   Ahmed Nassar ([ahn@zurich.ibm.com](mailto:ahn@zurich.ibm.com))
*   Peter Staar ([taa@zurich.ibm.com](mailto:taa@zurich.ibm.com))

ArXiv 링크: [https://arxiv.org/abs/2206.01062](https://arxiv.org/abs/2206.01062)

**인용 (Citation):**

```bibtex
@article{doclaynet2022,
  title = {DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis},
  doi = {10.1145/3534678.353904},
  url = {https://arxiv.org/abs/2206.01062},
  author = {Pfitzmann, Birgit and Auer, Christoph and Dolfi, Michele and Nassar, Ahmed S and Staar, Peter W J},
  year = {2022}
}
```

---

**참고:** 이 문서는 제공된 HTML 파일의 핵심 내용(README)을 Markdown 형식으로 번역하고 재구성한 것입니다. GitHub 페이지의 UI 요소(버튼, 메뉴 등) 및 동적 정보(실시간 커밋, 기여자 목록 등)는 제외되었습니다. 데이터셋 다운로드 링크 및 통계(Stars, Forks 등)는 HTML 로드 시점의 정보일 수 있습니다. 최신 정보는 원본 GitHub 저장소를 참조하세요.

