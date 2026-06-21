# su26-ai301-contribution
# Phase 1

I'd like to work on adding ECE or another metric to the Ignite Library.
## Contribution [#1]: Adding ECE, MRR or another Recommender System Metric to the Ignite Library.

**Contribution Number:** [1] 

**Student:** Sheetal Sattiraju 

**Issue:** https://github.com/pytorch/ignite/issues/2631 

**My Comment on the Issue:** Hello! I'm a student in the CodePath AI301 program. I'd like to try to contribute to this ticket. While reviewing the existing metrics in Ignite, I noticed that Expected Calibration Error (ECE) does not appear to be available in the main branch.

 I also came across issue #1009 but I wasn't sure whether ECE was implemented in another branch or if the work was never merged. Additionally, I looked at #2843 and wanted to ask whether that work regarding MRR is still active or completed. 

If MRR, ECE and Binary ECE are already being worked on/ complete, are there any remaining calibration metrics or recommender-system metrics from this issue that still need contributors? Let me know, thank you!

**Maintainer replied ✅**

---

### Why I Chose This Issue

I am interested in contributing to the PyTorch Ignite metrics module because it aligns with several of my learning goals: contributing to an open-source project, strengthening my machine learning knowledge, applying mathematical concepts, and improving my Python programming skills. While pursuing my Master's degree at San Jose State University, I have worked with evaluation metrics such as accuracy, precision, recall, and mean squared error, as well as recommender system specific metrics including Precision@K, Recall@K, MAP@K, NDCG@K, MRR. While exploring Ignite's existing metrics and open issues, I noticed the lack of calibration metrics in the main branch and an open issue related to expanding recommender-system metrics, which immediately caught my interest.

This issue is a good fit for my current skill set because it combines mathematical reasoning with Python implementation. Although I have experience using these metrics in coursework and projects, I have never implemented one within a production-quality open-source library. Through this contribution, I hope to gain experience reading a large codebase, writing tests, collaborating with maintainers, and deepening my understanding of how machine learning evaluation metrics are implemented in practice.

---

### Understanding the Issue

#### Problem Description

Ignite already provides a lot of metrics, as well as some specific recommender system evaluation metrics, including NDCG@K and Hit Rate in the main branch. However, the issue indicates that additional recommendation metrics and calibration are still missing from the library.

Adding the remaining metrics would make Ignite evaluation metrics more complete and provide users with a wider range of options for measuring model performance. Different metrics capture different aspects of recommendation quality, so having a broader set of metrics is crucial.

#### Expected Behavior

Adding a new metric in Ignite typically involves implementing a Metric class that integrates with the framework’s training and evaluation loop. Rather than computing the metric over the entire dataset at once, the implementation needs to support incremental updates over batches, accumulate necessary state during an epoch, and then compute the final result at the end. Need to also handle edge cases such as empty predictions or missing positive samples, and ensuring compatibility with distributed training setups.

In practice, this means defining how the metric processes model outputs and ground-truth labels in the update step, how intermediate results are stored, and how the final value is computed in the compute step. The implementation must also follow Ignite’s existing design patterns so it can be integrated easily along with appropriate unit tests and documentation to verify correctness and usability.


# Phase II

## Reproduction & Enviorment Setup
As the issue I selected is not a bug, but to implement a recommender system metric or calibration metric. This can be done in a Jupyter or any Python environment. The github issue calls for enchancement help needed. The environment set up is as follows:

```
pip install pytorch-ignite torch

from typing import Callable
import torch
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce
```

## Solution Plan:
The solution should be in one file, named chosen_metric.py and if merged, would be located in ignite/metrics/rec_sys/chosen_metric.py or a new folder ignite/metrics/calibration_metrics/chosen_metric.py. No other files should be modified. 

1. Map out how the metric works mathematically.
2. Write out the code for the metric.
3. Verify the code using ignite custom libraries and outside libraries.
4. Provide documentation for why the code works.
5. Document the tests and attach results in the github issue.

# Phase III
## Implementation Notes:
For Week 3, I decided to work on MRR (Rec Sys) Metric. 
Some notes on the math behind the metric:
* Mean Reciprocal Rank (MRR) measures how high the first relevant item appears in the recommendation list. 
* Basically, if the first relevant item appears near the top of the recommendation list, the model should receive a high score.
* But, if the first relevant item appears lower in the ranking, the score should decrease.
* Ranges from 0.0 (worst) to 1.0 (best)
Some notes on a prior PR attempt of creating MRR:
* Back in 2023, someone worked on an implementation but the maintainer noted parts of their code were wrong and they would come back when they understood the math better.
* Since then, maintainers and others have added a HitRate and NDCG@K implementation.
* Using these implementations, we can extend the logic to MRR.
* Their current metrics follow the pattern of filtering invalid users, computing a per-batch metric contribution for each top_k value, accumulating running sums & returning the average over all users. The prior implementation stored all relevance vectors (in PR #2843.)

**Base skeleton logic:**
1. rank items by prediction score
2. get relevance labels in ranked order
3. for each k:
      find first relevant item within top-k
      compute reciprocal rank
      add to running sum
return sum_mrr_per_k / num_examples

Link -->

I plan to implement a proper testing file in Week 4, but I did run some basic test cases in Colab.
