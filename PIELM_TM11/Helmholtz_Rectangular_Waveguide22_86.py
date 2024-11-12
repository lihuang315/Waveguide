import torch
import math,time
import numpy as np
from numpy import genfromtxt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.linalg import lstsq
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import lsqr, lsmr
#from torch_sparse_solve import solve

#PyTorch random number generator
torch.manual_seed(10)

# Random number generators in other libraries
np.random.seed(10)


device = torch.device('cpu')
print('Running on CPU!')


#a = 22.86*1.0e-3
#b = 10.16*1.0e-3
a = 22.86
b = 10.16
a_1 = 1.0/a
a_2 = 1.0/b

a_mid = a/2
b_mid = b/2

k2 = (a_1 * np.pi)**2 + (a_2 * np.pi)**2

#parameter
B=2.8e10
D0=1.3e-9
kB=1.38e-23
ee=1.609e-19
Ea=0.83*ee
T=373
Da=D0*np.exp(-Ea/(kB*T))
Omega=1.182e-29
rou=1.95e-8
Ze=1
kappa=Da*B*Omega/(kB*T)

sigma_std=1.0

start = time.time()

#Neural Network
class SLP(torch.nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size):
        super(SLP, self).__init__()
        #定义了一个权重矩阵,torch.nn.Parameter 将其标记为模型的参数，使得在训练过程中可以自动更新
        self.weights=torch.nn.Parameter(torch.Tensor(input_size, output_size))
        #定义了一个偏置向量，同样是神经网络的可学习参数
        self.bias = torch.nn.Parameter(torch.Tensor(output_size))
        #调用 reset_parameters 方法，用于初始化权重和偏置的数值
        self.reset_parameters()

    def reset_parameters(self):
        self.weights.data.uniform_(-1.0,1.0)  #将数值初始化为在区间[-1, 1]均匀分布的值
        self.bias.data.uniform_(-1.0,1.0)
        # self.weights.data.normal_(0,0.577)  #使用正态分布初始化权重矩阵和偏差
        # self.bias.data.normal_(0,0.577)
     

    def forward(self, input):
        activation = torch.nn.Tanh()
        y =torch.matmul(input, self.weights)+self.bias
        y = activation(y)

        return y

Num_n = 101
Net=SLP(2,Num_n)


#start = time.time()
#data
#PDE

#Num_g =np.array(60*L*10e4).astype(np.int32)

Num_g = 400

print('Num_g=',Num_g)



#X_g = (np.random.rand(Num_g,2)*[0.998,0.998]+[0.001, 0.001])*[L,t[1]-t[0]]+[0, t[0]]
#X_g = np.random.uniform(low=[0, 0], high=[L, t[1]], size=(Num_g, 2))
#X_g = np.random.uniform(low=[0.00001*a, 0.00001*b], high=[a, b], size=(Num_g, 2))
X_g = np.random.uniform(low=[0, 0], high=[a, b], size=(Num_g, 2))
X_input = X_g


#bc

Num_bc_a = 100
Num_bc_b = 50

#左边界
X_b_left = np.hstack((np.zeros((Num_bc_b,1)) ,(np.random.rand(Num_bc_b,1))*b ))
#右边界
X_b_right = np.hstack(((np.ones((Num_bc_b,1)))*a,(np.random.rand(Num_bc_b,1))*b))
#下边界
X_b_down = np.hstack(((np.random.rand(Num_bc_a,1))*a,(np.zeros((Num_bc_a,1)))))
#X_b_down = np.hstack(((np.random.rand(Num_bc_a,1)))*a,np.zeros((Num_bc_a,1)))
#上边界
X_b_up = np.hstack(((np.random.rand(Num_bc_a,1))*a,(np.ones((Num_bc_a,1)))*b))


#X_b=np.hstack((np.zeros((Num_bt,1)),np.random.rand(Num_bt,1)*(t[1]-t[0])+t[0]))
#X_b_t=np.hstack((np.ones((Num_bt,1))*L,np.random.rand(Num_bt,1)*(t[1]-t[0])+t[0]))
#X_b = np.vstack((X_b,X_b_t))
X_b = X_b_left
X_b = np.vstack( (X_b, X_b_right) )
X_b = np.vstack( (X_b, X_b_down) )
X_b = np.vstack( (X_b, X_b_up) )

Num_b = 2*(Num_bc_a + Num_bc_b)

X_input = np.vstack((X_input,X_b))


#ic
Num_i = 1
X_i = np.array([[a_mid, b_mid]])

X_input = np.vstack((X_input,X_i))

X_mean = np.mean(X_input,axis=0) #X_input 每一列的均值
X_std = np.std(X_input,axis=0)  #X_input 每一列的标准差

# normalization
X_g_norm = (X_g - X_mean) / X_std

X_b_norm = (X_b - X_mean) / X_std

X_i_norm = (X_i - X_mean) / X_std



np.savetxt('X_g.txt',X_g, delimiter = ',')

np.savetxt('X_b.txt',X_b, delimiter = ',')

np.savetxt('X_i.txt',X_i, delimiter = ',')



######  3D 显示点坐标  ###############
ax = plt.axes(projection='3d')


ax.scatter3D(X_g[:, 0], X_g[:, 1], c='green', label='X_g')
ax.scatter3D(X_b[:, 0], X_b[:, 1], c='blue', label='X_b')
ax.scatter3D(X_i[:, 0], X_i[:, 1], c='red', label='X_i')

ax.legend()  # 添加图例
#ax.scatter3D(X_g[:,0], X_g[:,1],  cmap='Greens')

#ax.scatter3D(X_b[:,0], X_b[:,1],  cmap='Greens')

#ax.scatter3D(X_i[:,0], X_i[:,1],  cmap='Greens')

plt.show()
################################################

##########  2-D图像  ############


# 绘制二维平面图
plt.scatter(X_g[:, 0], X_g[:, 1], c='green', label='X_g')
plt.scatter(X_b[:, 0], X_b[:, 1], c='blue', label='X_b')
plt.scatter(X_i[:, 0], X_i[:, 1], c='red', label='X_i')

plt.legend(loc='center', bbox_to_anchor=(0.5, 0.1), ncol=3)  # 水平放置，一行
#plt.legend()  # 添加图例

plt.xlabel('X-axis')  # 添加 x 轴标签
plt.ylabel('Y-axis')  # 添加 y 轴标签

plt.title('Scatter Plot')  # 添加图标题

plt.show()  # 显示图形



###########

Num_p = Num_g+Num_b+Num_i

AMatrix = torch.zeros((Num_p,Num_n))
BVector = torch.zeros((Num_p,1))


#pde
counter = 0

a=torch.Tensor(X_g_norm)
a=a.to(device=device)
a.requires_grad_()
b=Net(a)
A_pde = torch.zeros(Num_g,Num_n)

for j in range(Num_n):
    u=b[:,j:j+1]
    dx=torch.autograd.grad(outputs=u,inputs=a, grad_outputs=torch.ones_like(u), create_graph=True)[0][:,0:1]
    dy=torch.autograd.grad(outputs=u,inputs=a, grad_outputs=torch.ones_like(u), create_graph=True)[0][:,1:2]
    dxx=torch.autograd.grad(outputs=dx,inputs=a, grad_outputs=torch.ones_like(dx), create_graph=True)[0][:,0:1]
    dyy=torch.autograd.grad(outputs=dy, inputs=a, grad_outputs=torch.ones_like(dy), create_graph=True)[0][:,1:2]


    #A_pde[:, j:j + 1] = dxx + dyy + k2 * u
    A_pde[:, j:j + 1] = dxx * sigma_std / (X_std[0]) ** 2 + dyy * sigma_std / (X_std[1]) ** 2 + k2 * u* sigma_std

    #A_pde[:,j:j+1] = (dt*sigma_std/X_std[1]-kappa*dxx*sigma_std/(X_std[0])**2)*1e2


AMatrix[counter:counter+Num_g,:]=A_pde
#print('A_pde = ', A_pde[:,0])
counter = counter+Num_g
    
#bc
c=torch.Tensor(X_b_norm)
#c=c.to(device=device)
#c.requires_grad_()
#d=Net(c)
#把c分成两个：c_lr,c_du
c_lr = c[:2 * Num_bc_b, :].clone()
c_du = c[2 * Num_bc_b:, :].clone()
c_lr=c_lr.to(device=device)
c_lr.requires_grad_()
d_lr = Net(c_lr)
c_du=c_du.to(device=device)
c_du.requires_grad_()
d_du = Net(c_du)
#把d分为两种边界条件：d_lr,d_du
#d_lr = d[:2 * Num_bc_b, :].clone()
#d_du = d[2 * Num_bc_b:, :].clone()


print('c.shape=',c.shape)
#print('d.shape=',d.shape)
print('d_lr.shape=',d_lr.shape)
print('d_du.shape=',d_du.shape)

#A_b = torch.zeros(Num_b, Num_n)
# 左右边界条件
A_b_lr = torch.zeros(2*Num_bc_b, Num_n)
A_b_du = torch.zeros(2*Num_bc_a, Num_n)
#处理左右边界条件!!!!!!!!!!!!

for j in range(Num_n):
    bx=d_lr[:,j:j+1]
    #bdx=torch.autograd.grad(outputs=bx,inputs=c_lr, grad_outputs=torch.ones_like(bx), create_graph=True)[0][:,0:1]
    A_b_lr[:,j:j+1] = bx *sigma_std#/X_std[0]
    

AMatrix[counter:counter+2*Num_bc_b,:]=A_b_lr #* 1000
#BVector[counter:counter+Num_b,0:1] = torch.Tensor(-np.ones((Num_b,1))*J*ee*Ze/Omega*rou/1e11)

counter = counter + 2*Num_bc_b

#处理上下边界条件bc_down_up_dy=0
for j in range(Num_n):
    by = d_du[:, j:j + 1]
    #bdy = torch.autograd.grad(outputs=by, inputs=c_du, grad_outputs=torch.ones_like(by), create_graph=True)[0][:, 1:2]
    A_b_du[:, j:j + 1] = by *sigma_std  #/X_std[1]

AMatrix[counter:counter + 2 * Num_bc_a, :] = A_b_du #* 1000
# BVector[counter:counter+Num_b,0:1] = torch.Tensor(-np.ones((Num_b,1))*J*ee*Ze/Omega*rou/1e11)

counter = counter + 2 * Num_bc_a

#print('A_b_du = ', A_b_du[:,0])

# ic
e=torch.Tensor(X_i_norm)
e=e.to(device=device)
f=Net(e)

AMatrix[counter:counter+Num_i, :] = f * sigma_std
BVector[counter:counter+Num_i, 0:1] = torch.Tensor(np.ones((Num_i,1)))

#print('f_initial = ', f)

counter=counter+Num_i

start_time = time.time()

AMatrix = AMatrix.type(torch.DoubleTensor)
BVector = BVector.type(torch.DoubleTensor)
AT=torch.transpose(AMatrix,0,1)
Axx = torch.matmul(AT,AMatrix).detach().numpy()
Bxx = torch.matmul(AT,BVector).detach().numpy()

det_Axx = np.linalg.det(Axx)


np.savetxt('AMatrix.txt',AMatrix.detach().numpy(), delimiter = ',')
np.savetxt('BMatrix.txt',BVector.detach().numpy(), delimiter = ',')

np.savetxt('Axx.txt',Axx, delimiter = ',')
np.savetxt('Bxx.txt',Bxx, delimiter = ',')

# 检查行列式是否为零
if det_Axx != 0:
    print("Axx 是可逆的")
    # 在这里你可以继续求解线性方程组
    p = np.linalg.solve(Axx, Bxx)
else:
    print("Axx 不可逆，无法使用 np.linalg.solve 求解线性方程组")
    # 在这里你可能需要考虑其他解决方案，比如最小二乘法或其他方法
    p, residuals, rank, singular_values = np.linalg.lstsq(Axx, Bxx, rcond=None)

#p = np.linalg.solve(Axx,Bxx)



#print('ppppp=',p)
#############到这里了
data = genfromtxt('exact_mm100_TM11.txt', delimiter=',') #用于读取名为val1.txt的文本文件中的数据

X_input_norm = (data[:,0:2]-X_mean)/X_std

sigma = torch.matmul(Net(torch.Tensor(X_input_norm)),torch.Tensor(p))*sigma_std

elapsed = time.time() - start_time
print('Training time: %.2f' % (elapsed))


#################
np.savetxt('exact_u_TM11.txt',data[:,2], delimiter = ',')
np.savetxt('PIELM_TM11.txt',sigma.detach().numpy(), delimiter = ',')

# 将exact.txt的前两列和sigma的值合并成一个新的数组
result_data = np.column_stack((data[:, 0], data[:, 1], sigma.detach().numpy()))

# 将结果保存到'PIELM_2.txt'
np.savetxt('PIELM_TM11_quan.txt', result_data, delimiter=',')

ax = plt.axes(projection='3d')
ax.scatter3D(data[:,0], data[:,1], sigma.detach().numpy(),  s=1, c='tab:blue')
ax.scatter3D(data[:,0], data[:,1], data[:,2], s=1,  c='tab:orange')

plt.show()

#######################

###########################################################

###################################################
##################################################

import numpy as np
import matplotlib.pyplot as plt

# 假设 data[:, 0] 和 data[:, 1] 定义了一个网格
#X, Y = np.meshgrid(data[:, 0], data[:, 1])
X = data[:, 0].reshape(100,100)
Y = data[:, 1].reshape(100,100)
# 解析解
Z_exact = data[:, 2].reshape(100,100)

# 预测值
Z_pred = result_data[:, 2].reshape(100,100)

# 误差图
Z_error = np.abs(Z_exact - Z_pred)

# 绘图
fig_1 = plt.figure(1, figsize=(18, 5))

# 解析解的热图
plt.subplot(1, 3, 1)
plt.pcolor(X, Y, Z_exact, cmap='jet')
plt.xlabel(r'$x(mm)$', fontsize=18)
plt.ylabel(r'$y(mm)$', fontsize=18)
plt.title('Analytical result', fontsize=18)
plt.colorbar()
#plt.grid(True)

# 预测值的热图
plt.subplot(1, 3, 2)
pcm = plt.pcolor(X, Y, Z_pred, cmap='jet')
plt.xlabel(r'$x(mm)$', fontsize=18)
plt.ylabel(r'$y(mm)$', fontsize=18)
plt.title('Predicted result by PIELM', fontsize=18)
#plt.grid(True)
plt.colorbar()

# 误差图
plt.subplot(1, 3, 3)
#pcm = plt.pcolor(X, Y, Z_error, cmap='jet', vmin=0, vmax=np.max(Z_error))
#pcm = plt.pcolor(X, Y, Z_error, cmap='jet', vmin=0, vmax=0.01)
pcm = plt.pcolor(X, Y, Z_error, cmap='jet')
plt.xlabel(r'$x(mm)$', fontsize=18)
plt.ylabel(r'$y(mm)$', fontsize=18)
plt.title('Absolute error', fontsize=18)
#plt.grid(True)
plt.colorbar()

plt.tight_layout()

plt.savefig('Helmholtz_rec_PIELM_TM11.png', dpi=1000, bbox_inches='tight')
plt.show()


