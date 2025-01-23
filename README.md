
# Transformer 성능 최적화 및 프로파일링

## 개요
이 레포지토리는 **4개의 Nvidia Geforce RTX 3090 GPU**를 장착한 서버 환경에서 Transformer 모델 학습을 최적화하고 프로파일링하는 데 중점을 둡니다. 제한된 하드웨어 환경에서 GPU를 최대한 활용하여 **대형 언어 모델(LLM) 학습**의 효율성을 극대화하고자 합니다. 또한, CUDA 스트림 및 병렬 처리 기술을 활용하여 성능 병목 현상을 완화하려는 시도를 포함하고 있습니다.

특히, **서울대학교**의 **유승주 교수님**의 **"Advanced Computer Architecture"** (2024-1)와 **이재진 교수님**의 **"High Performance Computing"** (2024-1) 수업에서 배운 고급 이론과 기술들을 실제 연구 환경에 적용하여, PyTorch의 **Distributed Data Parallel (DDP)**와 **NVIDIA Nsight**, **CUDA Stream**을 활용한 최적화 기법을 구현하였습니다.  

본 프로젝트는 Mixed Precision Training, 프로파일링 도구를 사용한 병목 분석, CUDA 기반 병렬 처리 등의 기술을 통해 LLM 학습에서 발생하는 성능 병목 현상을 해결하는 것을 목표로 합니다.
"Attention is All You Need" 논문의 번역 태스크를 수행하는 transformer 모델의 학습을 프로파일링 및 최적화합니다.

전체 [보고서](assets/report.md)를 확인하실 수 있습니다.

[nsight profiling](https://drive.google.com/file/d/13kcJfWAqyTKbEWjT2_hyaUWN_-7I0nZX/view?usp=sharing) 파일을 다운받으실 수 있습니다.

---

## 목표
- **최대 학습 효율 달성**: Mixed Precision Training, Nsight Profiling, Distributed Data Parallel(DDP) 등의 기법을 활용하여 Transformer 학습 성능을 최적화.
- **CUDA 활용**: CUDA Stream을 이용한 병렬 처리를 통해 모델 연산 효율성을 개선하고, GPU 간 통신 오버헤드를 최소화.
- **지식 활용**: 다음 과목에서 배운 고급 이론과 기술을 프로젝트에 적용:
  - **Advanced Computer Architecture** (서울대학교, 유승주 교수님, 2024-1).
  - **High Performance Computing** (서울대학교, 이재진 교수님, 2024-1).
- **병목 현상 해결**: Gradient 동기화, GPU 메모리 불균형, CUDA Stream 오버헤드 등 주요 문제를 분석하고 해결.

---

## 주요 기능
- **Mixed Precision Training**: FP16과 FP32 연산을 혼합하여 계산 시간을 줄이고 메모리 사용량을 최적화.
- **Nsight System Profiling**: AllReduce 통신 오버헤드와 같은 성능 병목 현상을 식별하고 해결.
- **Gradient 동기화 최적화**: PyTorch DDP에서 `no_sync()`를 활용하여 효율적으로 gradient 누적 수행.
- **CUDA Stream 병렬화**:
  - Query, Key, Value 계산에서 CUDA Stream을 이용한 병렬화 실험.
  - GPU 연산 중 동기화 오버헤드와 스트림 생성 비용 분석.
- **Loss 계산 개선**: 각 GPU에서 병렬로 손실 계산을 수행하여 GPU0의 메모리 부담을 분산.
- **AllReduce Bucketing**: Bucket 크기를 조정하여 통신 비용을 최소화하고 효율성을 향상.

---

## 환경
- **하드웨어**:  
  - 4개의 Nvidia Geforce RTX 3090 GPU.
  - 서버 제약으로 인해 batch size 및 대규모 모델 구성에 제한이 있음.
- **소프트웨어**:  
  - PyTorch와 DistributedDataParallel(DDP).
  - Nsight 프로파일링 도구를 사용하여 성능 분석.
  - CUDA 스트림을 활용한 병렬 처리 구현.

---

## 주요 기술
1. **Mixed Precision Training**:  
   - Automatic Mixed Precision(AMP)와 Dynamic Loss Scaling을 활용하여 학습 속도와 메모리 사용량 최적화.
2. **프로파일링 및 병목 분석**:  
   - Nsight를 사용하여 AllReduce 연산에서 발생하는 통신 오버헤드를 식별.
3. **CUDA Stream 병렬화**:  
   - Query, Key, Value 연산에서 CUDA Stream을 이용한 성능 실험.
   - 스트림 동기화 오버헤드와 메모리 액세스 패턴 분석.
4. **Loss 계산 최적화**:  
   - 손실 계산을 병렬화하여 메모리 사용을 분산하고 GPU0의 부담 감소.
5. **Bucket 크기 조정**:  
   - Gradient 통신 빈도를 줄이기 위해 bucket 크기를 증가.


---

## 향후 연구 방향
- **Fully Sharded Data Parallel (FSDP)** 기법을 탐구하여 더 높은 메모리 효율성 달성.
- **Load Balancing 기법**을 연구하여 GPU 간 자원 활용도를 개선.
- CUDA Stream 활용 시 스트림 생성 및 동기화 오버헤드 감소 방안 연구.
- Bucket 크기 최적화 및 더 큰 데이터셋에서 실험 (하드웨어 제약이 허용하는 범위 내에서).


