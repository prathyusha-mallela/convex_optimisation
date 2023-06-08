# hw3 convex optimization
# Import packages.
import cvxpy as cp
import numpy as np
from scipy.io import loadmat
import scipy.linalg
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

#A is label 1, B is label -1

def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

def projection_v1(vec1, n):
    vec2 = np.zeros_like(vec1)
    vec2[0:n, 0], vec2[n:, 0] = projection_simplex_sort(
        vec1[0:n, 0]), projection_simplex_sort(vec1[n:, 0])
    return vec2


def projection_v2(vec1, n, d):
    vec2 = np.zeros_like(vec1)
    vec2 = np.clip(vec1, np.min(vec1), d)
    vec2[0:n, 0], vec2[n:, 0] = projection_simplex_sort(
        vec1[0:n, 0]), projection_simplex_sort(vec1[n:, 0])
    return vec2


def plot_svm(A_test, B_test, A, B, u_proj, v_proj, title='SVM'):
    plt.figure()
    # train set label 1 in red
    plt.scatter(A[0,:],A[1,:], c='red')
    # train set label -1 in blue
    plt.scatter(B[0,:], B[1,:], c='blue')
    # train set A*u_proj label 1 in red
    plt.scatter((A@u_proj)[0], (A@u_proj)[1], c='orange', marker=4)
    plt.scatter((B@v_proj)[0], (B@v_proj)[1], c='cyan', marker=4)
    x_vals = np.array(plt.axis()[0:2])
    midpoint = 0.5*(A@u_proj+B@v_proj)
    slope = (A@u_proj-B@v_proj)[1]/(A@u_proj-B@v_proj)[0]
    slope = -1/slope
    intercept = midpoint[1]-slope*midpoint[0]
    y_vals = intercept + slope * x_vals
    #calculate accuracy
    test_labels_1 = np.sign(-A_test[1,:]+intercept+(slope*A_test[0,:])).astype(int)
    test_labels_2 = np.sign(-B_test[1,:]+intercept+(slope*B_test[0,:])).astype(int)
    y_true = [1]*A_test.shape[1]+[-1]*B_test.shape[1]
    y_pred = list(test_labels_1)+list(test_labels_2)
    print(title,"error",(1-accuracy_score(y_true, y_pred))*100)
    plt.plot(x_vals, y_vals, c='green')
    plt.title(label=title+'_train')
    plt.savefig(title+'_train.png')
    plt.close()
    plt.figure()
    # test set label 1 in red
    plt.scatter((A@u_proj)[0], (A@u_proj)[1], c='orange', marker=4)
    plt.scatter((B@v_proj)[0], (B@v_proj)[1], c='cyan', marker=4)
    plt.scatter(A_test[0,:], A_test[1,:], c='darkred')
    # test set label -1 in blue
    plt.scatter(B_test[0,:], B_test[1,:], c='darkblue')
    plt.plot(x_vals, y_vals, c='green')
    plt.title(label=title+'_test')
    plt.savefig(title+'_test.png')
    plt.close()

# Problem 1 with CVXPY

np.random.seed(412381)

epsilon = 0.0001
results = dict()  # exp, opt, iter, funval, optima
results['separable'] = dict()
results['overlap'] = dict()

arr1 = loadmat('separable_case/train_separable.mat')
A = arr1['A']
B = arr1['B']

arr2 = loadmat('separable_case/test_separable.mat')
X_test = arr2['X_test'].T
labels = (arr2['true_labels'].T)[:, 0]

#pdb.set_trace()
A_test = X_test[np.where(labels == 1)].T
B_test = X_test[np.where(labels == -1)].T

m = A.shape[0]
n = A.shape[1]

# Define and solve the CVXPY problem.
x = cp.Variable(n)
y = cp.Variable(n)

cost = cp.sum_squares(A @ x - B @ y)
constraints = [x >= 0, y >= 0, cp.sum(x) == 1, cp.sum(y) == 1]
prob = cp.Problem(cp.Minimize(cost), constraints)
start = time.time()
prob.solve()
time_taken = time.time()-start

results['separable']['cvxpy'] = dict()
results['separable']['cvxpy']['u'] = np.expand_dims(x.value, axis=1)
results['separable']['cvxpy']['v'] = np.expand_dims(y.value, axis=1)
results['separable']['cvxpy']['funval'] = cp.sum_squares(A @ x - B @ y).value
results['separable']['cvxpy']['iter'] = dict()
results['separable']['cvxpy']['iter']['time'] = np.array([time_taken])
results['separable']['cvxpy']['iter']['funval'] = np.array(
    [cp.sum_squares(A @ x - B @ y).value])
results['separable']['cvxpy']['iter']['u'] = np.array(
    [np.expand_dims(x.value, axis=1)])
results['separable']['cvxpy']['iter']['v'] = np.array(
    [np.expand_dims(y.value, axis=1)])


plot_svm(A_test, B_test, A, B, np.expand_dims(x.value, axis=1),
         np.expand_dims(y.value, axis=1), title='plots/CVXPY_p1')

# Projected Gradient

u = np.random.random((n, 1))
v = np.random.random((n, 1))
alpha = 0.0005
u_proj = np.zeros_like(u)
v_proj = np.zeros_like(v)
u_proj[:, 0] = projection_simplex_sort(u[:, 0])
v_proj[:, 0] = projection_simplex_sort(v[:, 0])
old_func_value = np.power(np.linalg.norm(A@u_proj-B@v_proj), 2)
time_list = list()
u_list = list()
v_list = list()
fun_list = list()
start = time.time()
for i in range(5000):
    old_func_value = np.power(np.linalg.norm(A@u_proj-B@v_proj), 2)
    grad_u = 2.0 * A.T @ A @ u_proj - 2.0 * A.T @ B @ v_proj
    grad_v = 2.0 * B.T @ B @ v_proj - 2.0 * B.T @ A @ u_proj
    # do grad
    u_proj = u_proj-alpha*grad_u
    v_proj = v_proj-alpha*grad_v
    # do projection
    u_proj[:, 0] = projection_simplex_sort(u_proj[:, 0])
    v_proj[:, 0] = projection_simplex_sort(v_proj[:, 0])
    # calculate function value
    new_func_value = np.power(np.linalg.norm(A@u_proj-B@v_proj), 2)
    u_list.append(u_proj)
    v_list.append(v_proj)
    fun_list.append(new_func_value)
    time_list.append(time.time()-start)
    if np.abs(old_func_value-new_func_value) < epsilon:
        break

plot_svm(A_test, B_test, A, B, u_proj, v_proj,
         title='plots/projected_gradient_p1')
# plot_loss()

results['separable']['projected_gradient'] = dict()
results['separable']['projected_gradient']['u'] = u_proj
results['separable']['projected_gradient']['v'] = v_proj
results['separable']['projected_gradient']['funval'] = new_func_value
results['separable']['projected_gradient']['iter'] = dict()
results['separable']['projected_gradient']['iter']['time'] = np.array(
    time_list)
results['separable']['projected_gradient']['iter']['funval'] = np.array(
    fun_list)
results['separable']['projected_gradient']['iter']['u'] = np.array(u_list)
results['separable']['projected_gradient']['iter']['v'] = np.array(v_list)


# Nesterov's accelerated Gradient

C = np.zeros((m, n*2))
C[0:m, 0:n] = A
C[0:m, n:] = -B
Hessian = 2.0 * C.T @ C
L = scipy.linalg.svdvals(Hessian)[0]
x0 = np.zeros((2*n, 1))
x1 = np.zeros((2*n, 1))
x0[0:n, 0], x0[n:, 0] = projection_simplex_sort(
    x0[0:n, 0]), projection_simplex_sort(x0[n:, 0])
x1[0:n, 0], x1[n:, 0] = projection_simplex_sort(
    x1[0:n, 0]), projection_simplex_sort(x1[n:, 0])
y1 = np.copy(x1)
a0 = 1
time_list = list()
u_list = list()
v_list = list()
fun_list = list()
start = time.time()
for i in range(1000000):
    old_func_value = np.power(np.linalg.norm(C@x1), 2)
    a1 = (1+np.sqrt(4*a0*a0+1))/2
    tr = (a0-1)/a1
    y1 = (1+tr)*x1 - tr*x0
    x0 = x1
    a0 = a1
    grad = 2.0 * C.T @ C @ y1
    x1 = y1 - (1/L)*grad
    x1[0:n, 0], x1[n:, 0] = projection_simplex_sort(
    x1[0:n, 0]), projection_simplex_sort(x1[n:, 0])
    new_func_value = np.power(np.linalg.norm(C@x1), 2)
    fun_list.append(new_func_value)
    u_list.append(x1[0:n])
    v_list.append(x1[n:])
    time_list.append(time.time()-start)
    if np.abs(old_func_value-new_func_value) < epsilon and i > 0:
        break

results['separable']['nesterov'] = dict()
results['separable']['nesterov']['u'] = x1[0:n]
results['separable']['nesterov']['v'] = x1[n:]
results['separable']['nesterov']['funval'] = np.power(np.linalg.norm(C@x1), 2)
results['separable']['nesterov']['iter'] = dict()
results['separable']['nesterov']['iter']['time'] = np.array(time_list)
results['separable']['nesterov']['iter']['funval'] = np.array(fun_list)
results['separable']['nesterov']['iter']['u'] = np.array(u_list)
results['separable']['nesterov']['iter']['v'] = np.array(v_list)

plot_svm(A_test, B_test, A, B, x1[0:n], x1[n:], title='plots/Nesterov_p1')

##ADMM

m = A.shape[0]
n = A.shape[1]
C = np.zeros((m, n*2))
C[0:m, 0:n] = A
C[0:m, n:] = -B
x0 = np.random.random((2*n, 1))
x1 = projection_v1(x0, n)
rho = 50
mu = x1-x0
inv_mat = np.linalg.inv(C.T@C+rho*np.eye(2*n))
a0 = 1
time_list = list()
u_list = list()
v_list = list()
fun_list = list()
start = time.time()
for i in range(100000):
    old_func_value = np.power(np.linalg.norm(C@x1),2)
    x0 = inv_mat@(rho*x1-mu)
    x1 = projection_v1(x0+(mu/rho),n)
    mu = mu + rho*(x0 - x1)
    new_func_value = np.power(np.linalg.norm(C@x1),2)
    u_list.append(x1[0:n])
    v_list.append(x1[n:])
    fun_list.append(new_func_value)
    time_list.append(time.time()-start)
    if np.abs(old_func_value-new_func_value) < epsilon and i > 1:
        break

results['separable']['admm'] = dict()
results['separable']['admm']['u'] = x1[0:n]
results['separable']['admm']['v'] = x1[n:]
results['separable']['admm']['funval'] = np.power(np.linalg.norm(C@x1), 2)
results['separable']['admm']['iter'] = dict()
results['separable']['admm']['iter']['time'] = np.array(time_list)
results['separable']['admm']['iter']['funval'] = np.array(fun_list)
results['separable']['admm']['iter']['u'] = np.array(u_list)
results['separable']['admm']['iter']['v'] = np.array(v_list)

#pdb.set_trace()


plot_svm(A_test, B_test, A, B, x1[0:n], x1[n:], title='plots/admm_p1')

# Problem 2 with CVXPY

arr1 = loadmat('overlap_case/train_overlap.mat')
A = arr1['A']
B = arr1['B']

arr2 = loadmat('overlap_case/test_overlap.mat')
X_test = arr2['X_test'].T
labels = (arr2['true_labels'].T)[:, 0]

A_test = X_test[np.where(labels == 1)].T
B_test = X_test[np.where(labels == -1)].T

d = 0.05

m = A.shape[0]
n = A.shape[1]

# Define and solve the CVXPY problem.
x = cp.Variable(n)
y = cp.Variable(n)

cost = cp.sum_squares(A @ x - B @ y)
constraints = [x >= 0, y >= 0, x <= d, y <= d, cp.sum(x) == 1, cp.sum(y) == 1]
prob = cp.Problem(cp.Minimize(cost), constraints)
start = time.time()
prob.solve()
time_taken = time.time()-start

results['overlap']['cvxpy'] = dict()
results['overlap']['cvxpy']['u'] = np.expand_dims(x.value, axis=1)
results['overlap']['cvxpy']['v'] = np.expand_dims(y.value, axis=1)
results['overlap']['cvxpy']['funval'] = cp.sum_squares(A @ x - B @ y).value
results['overlap']['cvxpy']['iter'] = dict()
results['overlap']['cvxpy']['iter']['time'] = np.array([time_taken])
results['overlap']['cvxpy']['iter']['funval'] = np.array(
    [cp.sum_squares(A @ x - B @ y).value])
results['overlap']['cvxpy']['iter']['u'] = np.array(
    [np.expand_dims(x.value, axis=1)])
results['overlap']['cvxpy']['iter']['v'] = np.array(
    [np.expand_dims(y.value, axis=1)])

plot_svm(A_test, B_test, A, B, np.expand_dims(x.value, axis=1),
         np.expand_dims(y.value, axis=1), title='plots/CVXPY_p2')

# Projected Gradient

u = np.random.random((n, 1))
v = np.random.random((n, 1))
u_proj = np.zeros_like(u)
v_proj = np.zeros_like(v)
u_proj = np.clip(u, np.min(u), d)
v_proj = np.clip(v, np.min(v), d)
u_proj[:, 0] = projection_simplex_sort(u[:, 0])
v_proj[:, 0] = projection_simplex_sort(v[:, 0])
old_func_value = np.power(np.linalg.norm(A@u_proj-B@v_proj), 2)
time_list = list()
u_list = list()
v_list = list()
fun_list = list()
start = time.time()
for i in range(50000):
    old_func_value = np.power(np.linalg.norm(A@u_proj-B@v_proj), 2)
    grad_u = 2.0 * A.T @ A @ u_proj - 2.0 * A.T @ B @ v_proj
    grad_v = 2.0 * B.T @ B @ v_proj - 2.0 * B.T @ A @ u_proj
    # do grad
    u_proj = u_proj-alpha*grad_u
    v_proj = v_proj-alpha*grad_v
    # do projection
    u_proj = np.clip(u_proj, np.min(u_proj), d)
    v_proj = np.clip(v_proj, np.min(v_proj), d)
    u_proj[:, 0] = projection_simplex_sort(u_proj[:, 0])
    v_proj[:, 0] = projection_simplex_sort(v_proj[:, 0])
    # calculate function value
    new_func_value = np.power(np.linalg.norm(A@u_proj-B@v_proj), 2)
    u_list.append(u_proj)
    v_list.append(v_proj)
    fun_list.append(new_func_value)
    time_list.append(time.time()-start)
    if np.abs(old_func_value-new_func_value) < epsilon:
        break
plot_svm(A_test, B_test, A, B, u_proj, v_proj,
         title='plots/projected_gradient_p2')

results['overlap']['projected_gradient'] = dict()
results['overlap']['projected_gradient']['u'] = u_proj
results['overlap']['projected_gradient']['v'] = v_proj
results['overlap']['projected_gradient']['funval'] = new_func_value
results['overlap']['projected_gradient']['iter'] = dict()
results['overlap']['projected_gradient']['iter']['time'] = np.array(time_list)
results['overlap']['projected_gradient']['iter']['funval'] = np.array(fun_list)
results['overlap']['projected_gradient']['iter']['u'] = np.array(u_list)
results['overlap']['projected_gradient']['iter']['v'] = np.array(v_list)

# Nesterov's accelerated Gradient

C = np.zeros((m, n*2))
C[0:m, 0:n] = A
C[0:m:, n:] = -B
Hessian = 2.0 * C.T @ C
L = scipy.linalg.svdvals(Hessian)[0]
x0 = np.zeros((2*n, 1))
x1 = np.zeros((2*n, 1))
x0 = np.clip(x0, np.min(x0), d)
x1 = np.clip(x1, np.min(x1), d)
x0[0:n, 0], x0[n:, 0] = projection_simplex_sort(
    x0[0:n, 0]), projection_simplex_sort(x0[n:, 0])
x1[0:n, 0], x1[n:, 0] = projection_simplex_sort(
    x1[0:n, 0]), projection_simplex_sort(x1[n:, 0])
y1 = np.copy(x1)
a0 = 1
time_list = list()
u_list = list()
v_list = list()
fun_list = list()
start = time.time()
for i in range(1000000):
    old_func_value = np.power(np.linalg.norm(C@x1), 2)
    a1 = (1+np.sqrt(4*a0*a0+1))/2
    tr = (a0-1)/a1
    y1 = (1+tr)*x1 - tr*x0
    x0 = x1
    a0 = a1
    grad = 2.0 * C.T @ C @ y1
    x1 = y1 - (1/L)*grad
    x1 = np.clip(x1, np.min(x1), d)
    x1[0:n, 0], x1[n:, 0] = projection_simplex_sort(
    x1[0:n, 0]), projection_simplex_sort(x1[n:, 0])
    new_func_value = np.power(np.linalg.norm(C@x1), 2)
    fun_list.append(new_func_value)
    u_list.append(x1[0:n])
    v_list.append(x1[n:])
    time_list.append(time.time()-start)
    if np.abs(old_func_value-new_func_value) < epsilon and i > 0:
        break

results['overlap']['nesterov'] = dict()
results['overlap']['nesterov']['u'] = x1[0:n]
results['overlap']['nesterov']['v'] = x1[n:]
results['overlap']['nesterov']['funval'] = np.power(np.linalg.norm(C@x1), 2)
results['overlap']['nesterov']['iter'] = dict()
results['overlap']['nesterov']['iter']['time'] = np.array(time_list)
results['overlap']['nesterov']['iter']['funval'] = np.array(fun_list)
results['overlap']['nesterov']['iter']['u'] = np.array(u_list)
results['overlap']['nesterov']['iter']['v'] = np.array(v_list)

plot_svm(A_test, B_test, A, B, x1[0:n], x1[n:], title='plots/Nesterov_p2')

m = A.shape[0]
n = A.shape[1]
C = np.zeros((m, n*2))
C[0:m, 0:n] = A
C[0:m, n:] = -B
x0 = np.random.random((2*n, 1))
x1 = projection_v2(x0, n, d)
rho = 50
mu = x1-x0
inv_mat = np.linalg.inv(C.T@C+rho*np.eye(2*n))
time_list = list()
u_list = list()
v_list = list()
fun_list = list()
start = time.time()
for i in range(100000):
    old_func_value = np.power(np.linalg.norm(C@x1),2)
    x0 = inv_mat@(rho*x1-mu)
    x1 = projection_v2(x0+(mu/rho),n, d)
    mu = mu + rho*(x0 - x1)
    new_func_value = np.power(np.linalg.norm(C@x1),2)
    u_list.append(x1[0:n])
    v_list.append(x1[n:])
    fun_list.append(new_func_value)
    time_list.append(time.time()-start)
    if np.abs(old_func_value-new_func_value) < epsilon and i > 1:
        break

results['overlap']['admm'] = dict()
results['overlap']['admm']['u'] = x1[0:n]
results['overlap']['admm']['v'] = x1[n:]
results['overlap']['admm']['funval'] = np.power(np.linalg.norm(C@x1), 2)
results['overlap']['admm']['iter'] = dict()
results['overlap']['admm']['iter']['time'] = np.array(time_list)
results['overlap']['admm']['iter']['funval'] = np.array(fun_list)
results['overlap']['admm']['iter']['u'] = np.array(u_list)
results['overlap']['admm']['iter']['v'] = np.array(v_list)

plot_svm(A_test, B_test, A, B, x1[0:n], x1[n:], title='plots/admm_p2')

plt.close('all')
color_dict = {'cvxpy': 'red', 'projected_gradient': 'green',
              'nesterov': 'blue', 'admm': 'magenta'}

for key in results.keys():
    for exp in results[key]:
        print(exp)
        if exp != 'cvxpy':
            plt.title(label=key+'_'+exp+'_iteration')
            plt.xlabel('iteration steps')
            plt.ylabel('function value')
            plt.plot(np.arange(results[key][exp]['iter']['funval'].shape[0]),
                     results[key][exp]['iter']['funval'], c=color_dict[exp.lower()])
            plt.savefig('plots/'+key+'_'+exp+'_iteration.png')
            plt.close()

    for exp in results[key]:
        if exp != 'cvxpy':
            plt.title(label=key+'_'+exp+'_time')
            plt.xlabel('time in seconds')
            plt.ylabel('function value')
            plt.plot(results[key][exp]['iter']['time'], results[key]
                     [exp]['iter']['funval'], c=color_dict[exp.lower()])
            plt.savefig('plots/'+key+'_'+exp+'_time.png')
            plt.close()
    
    plt.title("run time comparision "+key)
    plt.xlabel('Experiment (CVXPY,PG,Nesterov,ADMM)')
    plt.ylabel('Time Taken')
    plt.scatter([1,2,3,4],[results[key]['cvxpy']['iter']['time'][-1],results[key]['projected_gradient']['iter']['time'][-1],results[key]['nesterov']['iter']['time'][-1],results[key]['admm']['iter']['time'][-1]])
    plt.savefig('plots/comparison/'+key+'_time_comparision.png')
    plt.close()  

    plt.title("iteration comparision "+key)
    plt.xlabel('Experiment (CVXPY,PG,Nesterov,ADMM)')
    plt.ylabel('Iterations Taken')
    plt.scatter([1,2,3,4],[len(results[key]['cvxpy']['iter']['time']),len(results[key]['projected_gradient']['iter']['time']),len(results[key]['nesterov']['iter']['time']),len(results[key]['admm']['iter']['time'])])
    plt.savefig('plots/comparison/'+key+'_iter_comparision.png')
    plt.close()        

    print("Problem", key, " Time taken for CVXPY", results[key]['cvxpy']['iter']['time'][-1], " projected gradient ", results[key]['projected_gradient']
          ['iter']['time'][-1], " nesterov", results[key]['nesterov']['iter']['time'][-1], " admm", results[key]['admm']['iter']['time'][-1])
    print("Problem", key, " Optimal Function value for CVXPY", results[key]['cvxpy']['funval'], " projected gradient ", results[key]['projected_gradient']
          ['funval'], " nesterov", results[key]['nesterov']['funval'], " admm", results[key]['admm']['funval'])
    # print("Problem", key, " Optimal u for CVXPY", results[key]['cvxpy']['u'][:,0], " projected gradient ", results[key]['projected_gradient']
    #       ['u'][:,0], " nesterov", results[key]['nesterov']['u'][:,0], " admm", results[key]['admm']['u'][:,0]) 
    # print("Problem", key, " Optimal v for CVXPY", results[key]['cvxpy']['v'], " projected gradient ", results[key]['projected_gradient']
    #       ['v'][:,0], " nesterov", results[key]['nesterov']['v'][:,0], " admm", results[key]['admm']['v'][:,0])                   
