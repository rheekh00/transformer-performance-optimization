# Transformer 모델 최적화 보고서

---

## 1. 소개

- **프로젝트 제목:** Transformer 모델 프로파일링 및 성능 향상  
- **작성자:** 서울대학교 컴퓨터공학과 최적화 및 금융공학 연구실 2023-22317 이기훈  
- **목적:**  
  현재 Transformer 모델의 학습 성능을 최적화하고, 학습 속도를 개선함으로써 더 다양한 데이터에 대해 효과적으로 실험할 수 있는 환경 구축.

---

## 2. 현재 환경

- **사용 모델:** PyTorch의 **Annotated Transformer** 구현을 기반으로 한 Transformer.  <br><br>
- **훈련 환경:**
  - **분산 데이터 병렬화(DDP):** 4대의 Geforce RTX 3090 GPU 사용.
  - **데이터셋 크기:** 5M 샘플.
  - 한 Epoch 학습 시간: 80분 이상 소요.<br><br>

- **현재 방식:** **Standard Data Parallel (DP)**
  ![image](https://github.com/user-attachments/assets/31258efc-1512-4b2c-8fc4-0f2736200fc3)

  - **작동 방식:**  
    - 데이터셋을 여러 GPU로 균등하게 나누고, 각 GPU에서 모델 복사본을 사용해 데이터를 독립적으로 처리.  
    - 처리 결과를 GPU0에서 집계하여 손실(Loss)을 계산하고, Backward Pass를 통해 가중치 업데이트를 진행.<br><br>
  - **특징 및 한계:**  
    - **장점:** 구현이 간단하며, 소규모 데이터셋에 적합.  
    - **단점:** 모든 GPU가 동기화되며, 가중치 업데이트 시 GPU0의 통신 및 메모리 부담이 큼.
    - 통신 오버헤드가 크기 때문에 대규모 데이터와 모델에서는 비효율적.<br><br>

- **문제점:**
  - PyTorch의 높은 추상화로 인해 최적화 작업이 어려움.
  - 통신 병목 현상 및 GPU 메모리 불균형으로 인해 성능 저하.
  - 다양한 데이터와 모델 실험을 위해 학습 시간 단축 필요.<br><br>

---

## 3. 최적화 기술

### 3.1 혼합 정밀도 학습(Mixed Precision Training)

- **정의:**  
  FP16(16비트)와 FP32(32비트) 정밀도를 병행하여 사용하는 학습 방식.<br><br>

- **이점:**
  - 학습 속도 **최대 3배 증가**.
  - **메모리 사용량 감소**, 동일한 하드웨어에서 더 큰 배치 또는 모델 학습 가능.
  - **확장성 증가**, 복잡한 모델 학습 또는 대규모 데이터셋 처리 가능.<br><br>

- **작동 방식:**
  - **FP16 연산:** Forward 및 Backward Pass 대부분을 Half Precision(FP16)으로 수행.
  - **FP32 마스터 복사:** 중요한 연산(예: 가중치 업데이트)은 Single Precision(FP32)로 유지하여 학습 안정성 확보.<br><br>

- **PyTorch 구현:**
  - `torch.cuda.amp`의 `autocast()` 및 `GradScaler()` 사용.
  - 기존 코드와 호환성을 위해 부울(bool) 타입에 대한 수정 필요.<br><br>
```python
with autocast():  # Apply mixed precision
    # Use no_sync() context for all but the last accumulation step
    if (i + 1) % accum_iter != 0:
        with model.no_sync():  # Disable gradient synchronization
            out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
            scaler.scale(loss_node).backward()  # Accumulate gradients locally
    else:
        out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        scaler.scale(loss_node).backward()  # This backward pass synchronizes gradients

if (i + 1) % accum_iter == 0:
    scaler.step(optimizer)  # Update model parameters
    scaler.update()
    optimizer.zero_grad(set_to_none=True)  # Reset gradients
    n_accum += 1
    train_state.accum_step += 1
    scheduler.step()  # Adjust learning rate
```

---

### 3.2 Nsight 프로파일링
![image](https://github.com/user-attachments/assets/bd513b17-f18e-4978-981f-868b38ebe74b)
![image](https://github.com/user-attachments/assets/9d1eaab2-8cc5-4212-aa0d-d67ed1d79295)


- **결과 분석:**
  - CUDA 스트림의 **79%가 AllReduce_Sum** 및 **Broadcast** 연산에 사용됨.
  - GPU 메모리 전송(H2D, D2H)은 성능에 미미한 영향을 끼침.<br><br>

- **문제점:**  
  - 그래디언트 동기화(AllReduce) 중 발생하는 통신 오버헤드가 성능 병목 현상 유발.<br><br>

- **개선 방향:**  
  - **AllReduce** 통신 최적화로 병목 현상 감소.<br><br>

- [nsight_profile.nsys-rep](https://drive.google.com/file/d/13kcJfWAqyTKbEWjT2_hyaUWN_-7I0nZX/view?usp=sharing)

---

### 3.3 AllReduce 최적화
<div style="display: flex; justify-content: center; align-items: center;">
  <img src="https://github.com/user-attachments/assets/3a52df69-52b3-49f3-b542-db58ffe4cbec" alt="AllReduce" style="width: 45%; margin-right: 10px;">
  <img src="https://github.com/user-attachments/assets/4e55cad8-7871-4403-9a45-e0e69b5c5475" alt="Ring-AllReduce 2" style="width: 45%;">
</div>


- **문제:**  
  - 그래디언트 동기화로 인해 GPU 간 통신 비용 증가.<br><br>
- **해결 방안:**  
  - **Ring AllReduce:** GPU 간 통신 비용 감소.
  - PyTorch의 `no_sync()` 활용:
    - 중간 그래디언트 동기화를 건너뛰고 마지막 단계에서만 동기화 수행.<br><br>
-  **Pytorch "no_sync()"**:
  - Pytorch의 torch.nn.parallel.DistributedDataParallel에서는 모든 backward pass를 수행할 때마다 기본적으로 gradient all-reduce를 실행.
  - 이는 gradient accumulation을 사용해 훈련하는 경우 불필요한 synchronization을 야기.
  - 따라서 각 gpu가 locally하게 N-1번의 gradient accumulation iteration을 수행한 후, weight update가 수행될 때만 동기화할 수 있게 해야 함.
  - 대부분의 pytorch train loop 코드에서 빠트리는 지점.<br><br>

-  **Pytorch "no_sync()" 실험**:
```Python
for i, batch in enumerate(data_iter):
    with autocast():  # Apply mixed precision
        # Use no_sync() context for all but the last accumulation step
        if (i + 1) % accum_iter != 0:
            with model.no_sync():  # Disable gradient synchronization
                out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
                loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
                scaler.scale(loss_node).backward()  # Accumulate gradients locally
        else:
            out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
            scaler.scale(loss_node).backward()  # This backward pass synchronizes gradients

    if (i + 1) % accum_iter == 0:
        scaler.step(optimizer)  # Update model parameters
        scaler.update()
        optimizer.zero_grad(set_to_none=True)  # Reset gradients
        n_accum += 1
        train_state.accum_step += 1
        scheduler.step()  # Adjust learning rate

```
  - DDP의 no_sync() context를 통해 불필요한 all-reduce 통신을 제거.

  
<div style="display: flex; justify-content: center; align-items: center;">
  <img src="https://github.com/user-attachments/assets/b1e5b6cc-ab8e-42e1-a96d-6af86ddf34b7" alt="Training Time Consumption Comparison" style="width: 45%; margin-right: 10px;">
  <img src="https://github.com/user-attachments/assets/d41bf74f-a6b6-4f5e-8f3c-2d169513d46b" alt="Training Throughput Comparison" style="width: 45%;">
</div>
  
-  **결과:**  
    - 더 큰 배치 크기와 다수의 GPU 사용 시 "no_sync"로 인한 성능 향상 기대 가능.

---

### 3.4 CUDA Streams

- **목적:** Query, Key, Value 계산을 병렬화하여 성능 개선.
  - W_q, W_k, W_v를 통해 query, key, value를 만드는 과정.
  - Transformer의 구현 코드 중 parallel하게 계산될 수 있는 유일한 부분.
  - CudaStream을 이용해 parallelize하여 성능 향상을 기대

```Python
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        # 1) Do all the linear projections in batch from d_model => h x d_k
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
        stream3 = torch.cuda.Stream()

        mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        torch.cuda.synchronize()

        with torch.cuda.stream(stream1):
            query = self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        with torch.cuda.stream(stream2):
            key = self.linears[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        with torch.cuda.stream(stream3):
            value = self.linears[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        torch.cuda.synchronize()

        del stream1, stream2, stream3

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        del query
        del key
        del value
        return self.linears[-1](x)

```

- **실험 결과:**
<div style="display: flex; justify-content: center; align-items: center;">
  <img src="https://github.com/user-attachments/assets/df6ed21e-1323-47fa-a9c7-7eb91a7524d2" alt="image 1" style="width: 45%; margin-right: 10px;">
  <img src="https://github.com/user-attachments/assets/935b8f4b-5014-49c1-8393-b277fbdf01e2" alt="image 2" style="width: 45%;">
</div>

  - 현재 환경에서는 오히려 성능 저하 확인.
  - 스트림 생성 및 동기화 오버헤드가 주요 원인.
  - 대규모 배치 크기에서 효과가 있을 가능성 존재.

---

### 3.5 Loss 계산 최적화

- **Forward Pass**:
![image](https://github.com/user-attachments/assets/722c90ed-aa68-4779-a165-327c7b366575)

  - 각 GPU가 Data Scatter, Model Broadcast를 받은 뒤 Forward Pass를 통해 logits를 계산.
  - GPU0에서 logits를 Gather하여 Loss를 구함.
  -> GPU0 memory의 unbalance
<br>

- **Backward Pass**:
![image](https://github.com/user-attachments/assets/71be7fd0-645a-4a1a-a26c-3ddfeabadad4)

  - GPU0에서 계산된 Loss를 GPUs에 Scatter.
  - 각 GPU에서 backward로 grad를 구한 뒤 GPU0로 ReduceAll (Ring).
  - GPU0에서 weight update step이 이뤄짐.
  - (GPUs logits) => [ GPU0 Gather => Loss compute ] => (GPUs Scatter => backward grad) => [GPU0 ReduceAll => Update Weights]
<br>

- **Loss Computation in Forward Pass**:
![image](https://github.com/user-attachments/assets/c7e77fbc-d455-4418-a959-396f76a1bb7e)

  - 각 GPU에서 loss를 locally compute하고 reduction 수행.
  - 각 결과로 GPU0로 Gather하여 다시 reduction.
  - loss를 2번 reduction하여 loss computation 과정을 parallelize하는 효과.
  -> GPU0 memory의 부담 감소
<br>


- **문제:**  
  - Forward 및 Backward Pass에서 GPU0의 메모리 부담이 과도함.
- **해결 방안:**
  - 각 GPU에서 Loss를 **로컬 계산** 후, Reduction 수행.
  - Reduction 병렬화를 통해 GPU0의 메모리 부담 감소.
  
```Python
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask, loss_compute, labels, ntokens):
        """
        Take in and process masked src and target sequences.
        """
        logits = self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
        return loss_compute(logits, labels, ntokens)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

```

---

### 3.6 그래디언트 버킷팅(Gradient Bucketing)
![image](https://github.com/user-attachments/assets/31ccc725-cbaf-4d6c-a4e7-6afd4f5f59e4)

<br>

- **All Reduce in Backward:**  
  PyTorch DDP는 `backward()` 단계에서 AllReduce를 수행. 이는 GPU 간 네트워크 통신과 Backward GPU 연산을 오버랩(중첩)하기 위해 설계됨.  
  - **작동 원리:**  
    Backward 연산은 신경망의 뒤쪽 레이어부터 순서대로 진행. 연산이 끝난 레이어의 gradient는 나머지 Backward 연산과 동시에 통신(overlap)이 가능.
  - **제한점:**  
    `step()` 단계에서 AllReduce가 수행되면 이러한 병렬화 효과를 누릴 수 없음.<br><br>

- **Gradient Bucketing:**  
  PyTorch DDP에서는 gradient 통신이 **버킷(bucket)**이라는 단위로 이루어짐.
  - **Bucket 정의:**  
    Bucket은 통신을 위한 gradient 복사본을 저장하는 버퍼. DDP를 사용할 때 GPUs에 모델을 로드하면, 모델 크기의 약 2배에 해당하는 메모리를 차지하게 되는 이유가 bucket 때문.
  - **작동 원리:**  
    Bucket은 반드시 레이어 단위로 통신하지 않으며, bucket이 가득 차면 `ReduceAll` 연산을 수행.
  - **기본 설정:**  
    PyTorch DDP에서 기본 bucket 크기는 **25MB**로 설정. Gradient 크기가 클 경우, bucket 크기를 키워 통신 횟수를 줄여 성능 최적화 가능.<br><br>

- **개선 방안:**  
  - 버킷 크기(기본: 25MB)를 증가시켜 통신 횟수 감소.
  - PyTorch DDP 설정에서 `gradient_as_bucket_view=True` 활성화:
    - 그래디언트 복사 대신 버킷 직접 참조로 메모리 사용 최적화.gradient를 메모리에 복사하지 않고, bucket을 직접 참조하여 추가적인 메모리 최적화 제공.<br><br>

---

### 3.7 Distributed Data Parallel(DDP)와 메모리 사용

- **GPU 메모리 제한:**  
  현재 Transformer 모델에서 DDP를 수행할 때 GPU당 최대 batch size는 512 미만.  
  - **모델 사양:**  
    - Layer 수: 6
    - Vocabulary size: 30,000
    - `d_model`: 512
    - `dff`: 2048
    - Head 수: 8
  - **제한 이유:** GPU0의 메모리 불균형(memory unbalancing) 때문.

- **DDP와 메모리 사용:**  
  DDP를 사용하면, GPU들에 모델을 로드할 때 모델 크기의 2배에 해당하는 메모리를 사용.  
  - **원인:**  
    - DDP는 각 GPU에 "bucket"을 만들어 다른 GPU로부터 gradient를 모으는 데 사용.  
    - 따라서 DDP 사용 시 gradient의 복사본이 하나 더 생성되어 추가적인 메모리를 소모.



---
## 4. 분석

- **핵심 결과:**
  - 혼합 정밀도 학습(Mixed Precision Training)은 성능 개선에 효과적.
  - 통신 오버헤드가 현재 환경의 주요 병목 현상.
  - CUDA Streams는 소규모 배치 크기에서 유효하지 않음.

- **권장 사항:**
  - AllReduce 통신 빈도 최적화.
  - GPU 로드 밸런싱 및 Fully Sharded Data Parallel(FSDP) 도입 검토.
  - PyTorch 최신 버전 유지 및 문서 업데이트 지속 확인.

---

## 5. 결론

- 혼합 정밀도 학습과 AllReduce 최적화를 통해 학습 속도와 자원 사용량을 개선함.
- CUDA Stream 병렬화는 현재 환경에서 유용하지 않았음.
- **향후 과제:**
  - GPU 간 로드 밸런싱 개선.
  - 버킷 크기 최적화.
  - Fully Sharded Data Parallel(FSDP) 구현.

- PyTorch 최신 기능 Follow up: [PyTorch 공식 문서](https://pytorch.org/)

---

