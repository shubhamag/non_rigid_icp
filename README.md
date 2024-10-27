# Non-rigid ICP (Iterative Closest Points)

[UPDATE 10/26/2024] : After 5 years of intending to clean this code up, finally, will get some time to upload a rewrite with more step by step comments, and reference to equations in the original paper where possible.

A modified, robust version of non-rigid Iterative closest point algorithm for deforming meshes to fit noisy point clouds

Also contains nicp_meshes.py, which registers a template to another mesh, a slightly improved version of the method proposed in:
"Amberg, B., Romdhani, S., & Vetter, T. (2007, June). Optimal step nonrigid icp algorithms for surface registration. In 2007 IEEE Conference on Computer Vision and Pattern Recognition (pp. 1-8). IEEE."
