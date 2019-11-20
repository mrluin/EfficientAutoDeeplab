import torch

class MyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_, b_):
        with torch.enable_grad():
            a = a_.detach().requires_grad_()
            res = a**2
        ctx.save_for_backward(a, res)
        return res.detach()
    @staticmethod
    def backward(ctx, grad_out):
        a, res = ctx.saved_tensors
        gr, = torch.autograd.grad(res, a, grad_out, retain_graph=True)
        y_gr = torch.randn(2,2, dtype=torch.double)
        return gr, y_gr

x = torch.randn(2,2, requires_grad=True, dtype=torch.double)
y = torch.randn(2,2, requires_grad=True, dtype=torch.double)
#print(torch.autograd.gradcheck(MyFn.apply, (x,y,)))

output = MyFn.apply(x, y)
output.sum().backward()
print(x.grad, y.grad)