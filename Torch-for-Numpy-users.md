torch equivalents of numpy functions

### Types
| Numpy            | Torch |
| --------------------|:-------------:|
| np.ndarray       | torch.Tensor
| np.float32       | torch.FloatTensor
| np.float64       | torch.DoubleTensor
| np.int8          | torch.CharTensor
| np.uint8         | torch.ByteTensor
| np.int16         | torch.ShortTensor
| np.int32         | torch.IntTensor
| np.int64         | torch.LongTensor

### Constructors
#### Ones and zeros
| Numpy            | Torch |
| --------------------|:-------------:|
| np.empty([2,2]) | torch.Tensor(2,2)
| np.empty_like(x) | torch.Tensor(x:size())
| np.eye           | torch.eye
| np.identity      | torch.eye
| np.ones          | torch.ones
| np.ones_like     | torch.ones(x:size())
| np.zeros         | torch.zeros
| np.zeros_like    | torch.zeros(x:size())

#### From existing data
| Numpy            | Torch |
| --------------------|:-------------:|
| np.array([[1,2][3,4]])   | torch.Tensor({{1,2},{3,4}})
| np.ascontiguousarray(x)   | x:contiguous()
| np.copy(x)    | x:clone()
| np.fromfile(file) | torch.Tensor(torch.Storage(file))
| np.frombuffer | ???
| np.fromfunction | ???
| np.fromiter | ???
| np.fromstring | ???
| np.loadtxt | ???

#### Numerical Ranges
| Numpy            | Torch |
| --------------------|:-------------:|
| np.arange(10)    | torch.linspace(1,10,10)
| np.arange(2, 3, 0.1) | torch.linspace(2, 2.9, 10)
| np.linspace(1, 4, 6) | torch.linspace(1, 4, 6)
| np.logspace | torch.logspace

#### Building Matrices
| Numpy            | Torch |
| --------------------|:-------------:|
| np.diag | torch.diag
| np.tril | torch.tril
| np.triu | torch.triu

#### Attributes
| Numpy            | Torch |
| --------------------|:-------------:|
| x.shape | x:size()
| x.strides | x:stride()
| x.ndim | x:dim()
| x.data | x:data()
| x.size | x:nElement()
| x.size == y.size | x:isSameSizeAs(y)
| x.dtype | x:type()

#### Indexing
| Numpy            | Torch |
| --------------------|:-------------:|

#### Shape Manipulation
| Numpy            | Torch |
| --------------------|:-------------:|
| x.reshape | x:reshape
| x.resize | x:resize
| ?        | x:resizeAs
| x.transpose | x:t()
| x.flatten   | x:view(x:nElement())
| x.squeeze   | x:squeeze

#### Item selection and manipulation
| Numpy            | Torch |
| --------------------|:-------------:|
| np.take(a, indices) | a[indices]
| np.put  | ????
| x.repeat | x:repeatTensor
| np.choose | ???
| np.sort | sorted, indices = torch.sort(x, [dim])
| np.argsort | sorted, indices = torch.sort(x, [dim])
| np.nonzero | x:gt(0)

#### Calculation
| Numpy            | Torch |
| --------------------|:-------------:|
| ndarray.min | mins, indices = torch.min(dim)
| ndarray.argmin | mins, indices = torch.min(dim)
| ndarray.max | maxs, indices = torch.max(dim)
| ndarray.argmax | maxs, indices = torch.max(dim)
| ndarray.clip | 
| ndarray.round | 
| ndarray.trace | torch.trace
| ndarray.sum | torch.sum
| ndarray.cumsum | torch.cumsum
| ndarray.mean | torch.mean
| ndarray.std | torch.std
| ndarray.prod | torch.prod
| ndarray.cumprod | torch.cumprod
| ndarray.all | ???
| ndarray.any | ???

#### Arithmetic and comparison operations
| Numpy            | Torch |
| --------------------|:-------------:|
| ndarray.__lt__ | torch.lt
| ndarray.__le__ | torch.le
| ndarray.__gt__ | torch.gt
| ndarray.__ge__ | torch.ge
| ndarray.__eq__ | torch.eq
| ndarray.__ne__ | torch.ne