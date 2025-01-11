# 24Fall-CORE-EL

## Team VEGE菜菜
### Team Members
- **Xiaoyu Yi**(Peking University)
- **Shu Yang**(Peking University)
- **Chi Hsu Tsai**(Peking University)
- **Yu Chen**(Peking University)
### Tools
Our project is implemented in Python, and we mainly use the toolkits which is [EGG](https://github.com/facebookresearch/EGG). And we also have some modifications to the EGG toolkit to adapt to our task, so we will provide the modified version of the EGG site-packages in each implementation floder. Moreover, we also use wandb to monitor the training process.

---

### Project Introduction
we propose a task that is more suitable for emerging language exploration.

In an environment where multiple agents collaborate to solve problems, the efficiency of communication between agents becomes an issue of great concern to us. For example, in a resource grabbing task, when multiple agents work together to complete the task, in a resource grabbing environment, some resources may not be available or are useless for the task goal. Then when one of the agents discovers this resource, it should discard the information and provide information about other useful resources.

---
### Task Description
Based on the task background, we envision the following tasks, which is a simple implementation we used for exploration. On an $N \times M$ map, there are $K$ ($K \leq N \times M$) locations marked with a value of V. At the same time, all four sides of $F$ ($F \leq K$) positions are filled with value R, which means that this position is surrounded. The goal is to output the positions of those Ks that are not surrounded. On the $10 \times 10$ map, there are 2 locations marked as 1, and there are three combinations based on the two locations being surrounded. So, this is used as input, and the final output is a map of the same size as the input, and the positions that are not surrounded are marked as 1, and the others are marked as 0.

---
### Floder Structure

- `implement_size5&20`: The implementation of the task with the map size of 5x5 and 20x20. And it is our beginning implementation of the task, the model is more simple and the performance is not good. Moreover, we add PCA for visualization.
- `implement_v2.0`: The implementation of the task mainly with the map size of 10x10. The model is more complex and the performance is better than the previous one, the loss function is also improved, and we also add the metrix to evaluate the performance of the task.
- `implement_v2.1`: The implementation of the task mainly with the map size of 10x10. It is our final implementation of the task, everything is the same as the previous one, but change the loss function for more exploration.

Due to our time limitation, we just only provide the detailed information of the `implement_v2.1` floder. Therefore, if you want to know more about our project, please refer to the `implement_v2.1` floder.

---
### Other Information
If you have any questions, please contact us. We will be happy to help you. Thanks for your attention.