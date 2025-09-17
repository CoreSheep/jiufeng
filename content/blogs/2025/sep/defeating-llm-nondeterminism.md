---
title: "Defeating Nondeterminism in LLM Inference: A Breakthrough in Deep Learning System Reproducibility"
date: 2025-09-15T10:00:00+01:00
draft: false
tags: ["machine-learning", "llm", "determinism", "reproducibility", "research"]
categories: ["AI Research"]
description: "An in-depth analysis of how the Thinking Machines team solved the nondeterminism problem in large language model inference, achieving truly reproducible reasoning."
image: "/img/blog/llm-determinism.jpg"
---

# Defeating Nondeterminism in LLM Inference: A Breakthrough in Deep Learning System Reproducibility

## Introduction

**Reproducibility** stands as one of the fundamental pillars of scientific research. However, when dealing with large language models (LLMs), achieving reproducible results has proven extraordinarily challenging. Even when we set the temperature parameter to 0 (greedy sampling), which should theoretically produce deterministic results, LLM inference still exhibits nondeterministic behavior.

Recently, the Thinking Machines team published [groundbreaking research](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) that deeply investigates the root causes of nondeterminism in LLM inference and proposes effective solutions. This article provides an in-depth analysis of the core principles and methods of this research.

## The Nature of the Problem: Floating-Point Non-Associativity

### The "Original Sin" of Floating-Point Operations

To understand the root of nondeterminism, we must first grasp the concept of **floating-point non-associativity**. In floating-point arithmetic:

$$(a + b) + c \neq a + (b + c)$$

This seemingly simple mathematical property is actually the fundamental cause of nondeterminism in large language model inference.

```python
# Simple example of floating-point non-associativity
(0.1 + 1e20) - 1e20  # Result: 0
0.1 + (1e20 - 1e20)  # Result: 0.1
```

### Dynamic Precision and Information Loss

Floating-point systems balance numerical range and precision through "dynamic precision." When adding two floating-point numbers with different exponents, the system must discard some precision information:

- 1230 ($1.23 \times 10^2$) + 23.4 ($2.34 \times 10^1$) = 1253.4
- But due to maintaining only 3 digits of precision, the result is truncated to 1250 ($1.25 \times 10^2$)

This means that **every time floating-point numbers are added in different orders, completely different results may be obtained**.

## Limitations of Traditional Explanations

### Insufficiency of the "Concurrency + Floating-Point" Hypothesis

For a long time, the academic community has generally attributed the nondeterminism in LLM inference to the "concurrency + floating-point" hypothesis:

> Due to the parallel computing characteristics of GPUs, the completion order of different threads is nondeterministic, leading to inconsistent floating-point accumulation orders.

However, this research reveals the limitations of this hypothesis:

1. **GPU matrix multiplication is deterministic**: Even highly parallel matrix multiplication operations can produce bit-level identical results when repeatedly executed on the same data
2. **Not all concurrent operations lead to nondeterminism**: The key lies in specific implementation methods, not concurrency itself

## The Real Culprit: Batch Non-Invariance

### Core Discovery

The research team discovered that the true root of LLM inference nondeterminism is **batch non-invariance**. Specifically manifested as:

- The same data produces different numerical results under different batch sizes
- These differences accumulate and amplify during the inference process
- Ultimately leading to completely different output sequences

### Problems with Chunked Reduction Strategies

In attention mechanism calculations, when query length is small (such as during the decoding phase), chunked reduction strategies are needed to fully utilize GPU parallelism. The problem lies in:

```python
# Problem example: Dynamic chunking strategy
# KV length = 1000, requires 4 chunks
# Each core processes 250 elements
# But chunk count depends on batch size and query length
```

This dynamic chunking strategy breaks batch invariance because:
- Chunking strategy depends on the current number of queries being processed
- Different requests may trigger different chunking strategies
- Leading to differences in floating-point accumulation order

## Solution: Batch-Invariant Kernels

### Fixed-Size Chunking Strategy

The core solution proposed by the research team is to adopt a **fixed-size chunking strategy**:

```python
# Solution: Fixed-size chunking
# Regardless of KV length, each chunk is fixed at 256 elements
# KV length = 1000 → 3 chunks of 256 + 1 chunk of 232
# KV length = 512  → 2 chunks of 256
```

Advantages of this strategy:
- **Batch invariance**: Same reduction order is executed regardless of how many tokens are processed
- **Reproducibility**: Same input always produces same output
- **Performance preservation**: Still able to fully utilize GPU parallelism

### Implementation Details

The team achieved deterministic inference through the following technologies:

1. **torch.Library integration**: Non-invasive replacement of PyTorch operators
2. **FlexAttention backend**: Implementation based on vLLM's FlexAttention
3. **Batch-invariant kernels**: Specially designed kernels ensuring numerical stability

## Experimental Results and Verification

### Nondeterminism Level Assessment

Testing with the Qwen/Qwen3-235B model:
- **Traditional method**: 1000 identical prompts generated 80 different results
- **Deterministic method**: 1000 identical prompts generated completely identical results

Notably, even with the nondeterministic method, the first 102 tokens were completely identical, with differences beginning to appear from the 103rd token.

### Performance Impact

| Configuration | Time (seconds) |
|---------------|----------------|
| vLLM Default | 26 |
| Unoptimized Deterministic vLLM | 55 |
| Improved Attention Kernel | 42 |

While deterministic inference incurs some performance overhead, it remains within acceptable ranges.

### Breakthrough in Reinforcement Learning

More importantly, this research solves a critical problem in reinforcement learning:

- **Traditional problem**: Numerical differences between training and inference lead to "fake on-policy" reinforcement learning
- **Solution**: Deterministic inference makes true on-policy reinforcement learning possible
- **Verification results**: In RLVR experiments, the deterministic method achieved 0 KL divergence, indicating complete consistency between training and sampling policies

## Technical Implementation and Open Source Contributions

### Open Source Resources

The research team provided complete implementations:

- **Batch-invariant operations library**: [thinking-machines-lab/batch-invariant-ops](https://github.com/thinking-machines-lab/batch-invariant-ops)
- **vLLM deterministic mode examples**: Directly runnable code demonstrations

### Core Code Structure

```python
# Core idea of batch-invariant kernels
def batch_invariant_reduction(data, reduction_dim):
    # Fixed chunk size, not fixed chunk count
    fixed_chunk_size = 256
    chunks = split_into_fixed_size_chunks(data, fixed_chunk_size)
    
    # Ensure consistency of reduction order
    result = deterministic_reduce(chunks)
    return result
```

## Significance for AI Research

### Enhancement of Scientific Rigor

The value of this research lies not only in technical breakthroughs but also in its contribution to the scientific rigor of AI research:

1. **Reproducibility**: Researchers can completely reproduce experimental results
2. **Debugging capability**: Ability to precisely locate and fix numerical issues
3. **System understanding**: Deep understanding of the complexity of modern GPU computing systems

### Practical Application Value

- **Model deployment**: Ensuring consistent behavior in production environments
- **A/B testing**: Eliminating the impact of randomness on experimental results
- **Reinforcement learning**: Enabling true on-policy learning

## Conclusion and Outlook

The research by the Thinking Machines team reveals the true root of nondeterminism in LLM inference and provides practical solutions. This work not only solves technical problems but more importantly enhances the scientific rigor of the entire AI research field.

### Key Insights

1. **Don't accept "this is normal"**: When facing nondeterminism issues, we should dig deep into root causes
2. **Importance of systems thinking**: Understanding interaction effects in multi-layered abstract systems
3. **Integration of engineering and science**: Validating scientific hypotheses through engineering practice

### Future Directions

- **Performance optimization**: Further optimizing the performance of deterministic kernels
- **Broader applicability**: Extending methods to more model architectures
- **Standardization promotion**: Promoting deterministic inference as an industry standard

This research reminds us that in today's rapid AI development, we still need to maintain attention to fundamental issues and solve seemingly complex engineering problems through rigorous scientific methods. The implementation of deterministic inference is not only a technical achievement but also a persistence and practice of scientific methodology.

---

*References:*
- [He, Horace and Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference", Thinking Machines Lab: Connectionism, Sep 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [GitHub: batch-invariant-ops](https://github.com/thinking-machines-lab/batch-invariant-ops)