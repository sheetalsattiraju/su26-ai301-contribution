# su26-ai301-contribution
(If approved) I'd like to work on adding ECE or another metric to the Ignite Library.
# Contribution [#1]: Adding ECE or another Recommender System Metric to the Ignite Library.

**Contribution Number:** [1] 

**Student:** Sheetal Sattiraju 

**Issue:** https://github.com/pytorch/ignite/issues/2631 

**Status:** [Phase I] [In Progress]

---

## Why I Chose This Issue

I am interested in contributing to the PyTorch Ignite metrics module because it aligns with several of my learning goals: contributing to an open-source project, strengthening my machine learning knowledge, applying mathematical concepts, and improving my Python programming skills. While pursuing my Master's degree at San Jose State University, I have worked with evaluation metrics such as accuracy, precision, recall, and mean squared error, as well as recommender system specific metrics including Precision@K, Recall@K, MAP@K, NDCG@K, MRR. While exploring Ignite's existing metrics and open issues, I noticed the lack of calibration metrics in the main branch and an open issue related to expanding recommender-system metrics, which immediately caught my interest.

This issue is a good fit for my current skill set because it combines mathematical reasoning with Python implementation. Although I have experience using these metrics in coursework and projects, I have never implemented one within a production-quality open-source library. Through this contribution, I hope to gain experience reading a large codebase, writing tests, collaborating with maintainers, and deepening my understanding of how machine learning evaluation metrics are implemented in practice.

---

## Understanding the Issue

### Problem Description

Ignite already provides a lot of metrics, as well as some specific recommender system evaluation metrics, including NDCG@K and Hit Rate in the main branch. However, the issue indicates that additional recommendation metrics and calibration are still missing from the library.

Adding the remaining metrics would make Ignite evaluation metrics more complete and provide users with a wider range of options for measuring model performance. Different metrics capture different aspects of recommendation quality, so having a broader set of metrics is crucial.

### Expected Behavior

Adding a new metric in Ignite typically involves implementing a Metric class that integrates with the framework’s training and evaluation loop. Rather than computing the metric over the entire dataset at once, the implementation needs to support incremental updates over batches, accumulate necessary state during an epoch, and then compute the final result at the end. Need to also handle edge cases such as empty predictions or missing positive samples, and ensuring compatibility with distributed training setups.

In practice, this means defining how the metric processes model outputs and ground-truth labels in the update step, how intermediate results are stored, and how the final value is computed in the compute step. The implementation must also follow Ignite’s existing design patterns so it can be integrated easily along with appropriate unit tests and documentation to verify correctness and usability.
