??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
;convolutional_autoencoder/convolution_encoder/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*L
shared_name=;convolutional_autoencoder/convolution_encoder/conv2d/kernel
?
Oconvolutional_autoencoder/convolution_encoder/conv2d/kernel/Read/ReadVariableOpReadVariableOp;convolutional_autoencoder/convolution_encoder/conv2d/kernel*&
_output_shapes
:*
dtype0
?
9convolutional_autoencoder/convolution_encoder/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9convolutional_autoencoder/convolution_encoder/conv2d/bias
?
Mconvolutional_autoencoder/convolution_encoder/conv2d/bias/Read/ReadVariableOpReadVariableOp9convolutional_autoencoder/convolution_encoder/conv2d/bias*
_output_shapes
:*
dtype0
?
=convolutional_autoencoder/convolution_encoder/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=convolutional_autoencoder/convolution_encoder/conv2d_1/kernel
?
Qconvolutional_autoencoder/convolution_encoder/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp=convolutional_autoencoder/convolution_encoder/conv2d_1/kernel*&
_output_shapes
: *
dtype0
?
;convolutional_autoencoder/convolution_encoder/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *L
shared_name=;convolutional_autoencoder/convolution_encoder/conv2d_1/bias
?
Oconvolutional_autoencoder/convolution_encoder/conv2d_1/bias/Read/ReadVariableOpReadVariableOp;convolutional_autoencoder/convolution_encoder/conv2d_1/bias*
_output_shapes
: *
dtype0
?
=convolutional_autoencoder/convolution_encoder/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*N
shared_name?=convolutional_autoencoder/convolution_encoder/conv2d_2/kernel
?
Qconvolutional_autoencoder/convolution_encoder/conv2d_2/kernel/Read/ReadVariableOpReadVariableOp=convolutional_autoencoder/convolution_encoder/conv2d_2/kernel*&
_output_shapes
: @*
dtype0
?
;convolutional_autoencoder/convolution_encoder/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*L
shared_name=;convolutional_autoencoder/convolution_encoder/conv2d_2/bias
?
Oconvolutional_autoencoder/convolution_encoder/conv2d_2/bias/Read/ReadVariableOpReadVariableOp;convolutional_autoencoder/convolution_encoder/conv2d_2/bias*
_output_shapes
:@*
dtype0
?
=convolutional_autoencoder/convolution_encoder/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*N
shared_name?=convolutional_autoencoder/convolution_encoder/conv2d_3/kernel
?
Qconvolutional_autoencoder/convolution_encoder/conv2d_3/kernel/Read/ReadVariableOpReadVariableOp=convolutional_autoencoder/convolution_encoder/conv2d_3/kernel*'
_output_shapes
:@?*
dtype0
?
;convolutional_autoencoder/convolution_encoder/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*L
shared_name=;convolutional_autoencoder/convolution_encoder/conv2d_3/bias
?
Oconvolutional_autoencoder/convolution_encoder/conv2d_3/bias/Read/ReadVariableOpReadVariableOp;convolutional_autoencoder/convolution_encoder/conv2d_3/bias*
_output_shapes	
:?*
dtype0
?
:convolutional_autoencoder/convolution_encoder/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	*K
shared_name<:convolutional_autoencoder/convolution_encoder/dense/kernel
?
Nconvolutional_autoencoder/convolution_encoder/dense/kernel/Read/ReadVariableOpReadVariableOp:convolutional_autoencoder/convolution_encoder/dense/kernel*
_output_shapes
:	?	*
dtype0
?
8convolutional_autoencoder/convolution_encoder/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8convolutional_autoencoder/convolution_encoder/dense/bias
?
Lconvolutional_autoencoder/convolution_encoder/dense/bias/Read/ReadVariableOpReadVariableOp8convolutional_autoencoder/convolution_encoder/dense/bias*
_output_shapes
:*
dtype0
?
<convolutional_autoencoder/convolution_decoder/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*M
shared_name><convolutional_autoencoder/convolution_decoder/dense_1/kernel
?
Pconvolutional_autoencoder/convolution_decoder/dense_1/kernel/Read/ReadVariableOpReadVariableOp<convolutional_autoencoder/convolution_decoder/dense_1/kernel*
_output_shapes
:	?*
dtype0
?
:convolutional_autoencoder/convolution_decoder/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*K
shared_name<:convolutional_autoencoder/convolution_decoder/dense_1/bias
?
Nconvolutional_autoencoder/convolution_decoder/dense_1/bias/Read/ReadVariableOpReadVariableOp:convolutional_autoencoder/convolution_decoder/dense_1/bias*
_output_shapes	
:?*
dtype0
?
=convolutional_autoencoder/convolution_decoder/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*N
shared_name?=convolutional_autoencoder/convolution_decoder/conv2d_4/kernel
?
Qconvolutional_autoencoder/convolution_decoder/conv2d_4/kernel/Read/ReadVariableOpReadVariableOp=convolutional_autoencoder/convolution_decoder/conv2d_4/kernel*'
_output_shapes
:?*
dtype0
?
;convolutional_autoencoder/convolution_decoder/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*L
shared_name=;convolutional_autoencoder/convolution_decoder/conv2d_4/bias
?
Oconvolutional_autoencoder/convolution_decoder/conv2d_4/bias/Read/ReadVariableOpReadVariableOp;convolutional_autoencoder/convolution_decoder/conv2d_4/bias*
_output_shapes	
:?*
dtype0
?
=convolutional_autoencoder/convolution_decoder/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*N
shared_name?=convolutional_autoencoder/convolution_decoder/conv2d_5/kernel
?
Qconvolutional_autoencoder/convolution_decoder/conv2d_5/kernel/Read/ReadVariableOpReadVariableOp=convolutional_autoencoder/convolution_decoder/conv2d_5/kernel*(
_output_shapes
:??*
dtype0
?
;convolutional_autoencoder/convolution_decoder/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*L
shared_name=;convolutional_autoencoder/convolution_decoder/conv2d_5/bias
?
Oconvolutional_autoencoder/convolution_decoder/conv2d_5/bias/Read/ReadVariableOpReadVariableOp;convolutional_autoencoder/convolution_decoder/conv2d_5/bias*
_output_shapes	
:?*
dtype0
?
=convolutional_autoencoder/convolution_decoder/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*N
shared_name?=convolutional_autoencoder/convolution_decoder/conv2d_6/kernel
?
Qconvolutional_autoencoder/convolution_decoder/conv2d_6/kernel/Read/ReadVariableOpReadVariableOp=convolutional_autoencoder/convolution_decoder/conv2d_6/kernel*'
_output_shapes
:?@*
dtype0
?
;convolutional_autoencoder/convolution_decoder/conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*L
shared_name=;convolutional_autoencoder/convolution_decoder/conv2d_6/bias
?
Oconvolutional_autoencoder/convolution_decoder/conv2d_6/bias/Read/ReadVariableOpReadVariableOp;convolutional_autoencoder/convolution_decoder/conv2d_6/bias*
_output_shapes
:@*
dtype0
?
=convolutional_autoencoder/convolution_decoder/conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *N
shared_name?=convolutional_autoencoder/convolution_decoder/conv2d_7/kernel
?
Qconvolutional_autoencoder/convolution_decoder/conv2d_7/kernel/Read/ReadVariableOpReadVariableOp=convolutional_autoencoder/convolution_decoder/conv2d_7/kernel*&
_output_shapes
:@ *
dtype0
?
;convolutional_autoencoder/convolution_decoder/conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *L
shared_name=;convolutional_autoencoder/convolution_decoder/conv2d_7/bias
?
Oconvolutional_autoencoder/convolution_decoder/conv2d_7/bias/Read/ReadVariableOpReadVariableOp;convolutional_autoencoder/convolution_decoder/conv2d_7/bias*
_output_shapes
: *
dtype0
?
=convolutional_autoencoder/convolution_decoder/conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=convolutional_autoencoder/convolution_decoder/conv2d_8/kernel
?
Qconvolutional_autoencoder/convolution_decoder/conv2d_8/kernel/Read/ReadVariableOpReadVariableOp=convolutional_autoencoder/convolution_decoder/conv2d_8/kernel*&
_output_shapes
: *
dtype0
?
;convolutional_autoencoder/convolution_decoder/conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*L
shared_name=;convolutional_autoencoder/convolution_decoder/conv2d_8/bias
?
Oconvolutional_autoencoder/convolution_decoder/conv2d_8/bias/Read/ReadVariableOpReadVariableOp;convolutional_autoencoder/convolution_decoder/conv2d_8/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
BAdam/convolutional_autoencoder/convolution_encoder/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*S
shared_nameDBAdam/convolutional_autoencoder/convolution_encoder/conv2d/kernel/m
?
VAdam/convolutional_autoencoder/convolution_encoder/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpBAdam/convolutional_autoencoder/convolution_encoder/conv2d/kernel/m*&
_output_shapes
:*
dtype0
?
@Adam/convolutional_autoencoder/convolution_encoder/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@Adam/convolutional_autoencoder/convolution_encoder/conv2d/bias/m
?
TAdam/convolutional_autoencoder/convolution_encoder/conv2d/bias/m/Read/ReadVariableOpReadVariableOp@Adam/convolutional_autoencoder/convolution_encoder/conv2d/bias/m*
_output_shapes
:*
dtype0
?
DAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *U
shared_nameFDAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/kernel/m
?
XAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpDAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/kernel/m*&
_output_shapes
: *
dtype0
?
BAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *S
shared_nameDBAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/bias/m
?
VAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpBAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/bias/m*
_output_shapes
: *
dtype0
?
DAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*U
shared_nameFDAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/kernel/m
?
XAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpDAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/kernel/m*&
_output_shapes
: @*
dtype0
?
BAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*S
shared_nameDBAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/bias/m
?
VAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpBAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/bias/m*
_output_shapes
:@*
dtype0
?
DAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*U
shared_nameFDAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/kernel/m
?
XAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpDAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/kernel/m*'
_output_shapes
:@?*
dtype0
?
BAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*S
shared_nameDBAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/bias/m
?
VAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpBAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/bias/m*
_output_shapes	
:?*
dtype0
?
AAdam/convolutional_autoencoder/convolution_encoder/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	*R
shared_nameCAAdam/convolutional_autoencoder/convolution_encoder/dense/kernel/m
?
UAdam/convolutional_autoencoder/convolution_encoder/dense/kernel/m/Read/ReadVariableOpReadVariableOpAAdam/convolutional_autoencoder/convolution_encoder/dense/kernel/m*
_output_shapes
:	?	*
dtype0
?
?Adam/convolutional_autoencoder/convolution_encoder/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/convolutional_autoencoder/convolution_encoder/dense/bias/m
?
SAdam/convolutional_autoencoder/convolution_encoder/dense/bias/m/Read/ReadVariableOpReadVariableOp?Adam/convolutional_autoencoder/convolution_encoder/dense/bias/m*
_output_shapes
:*
dtype0
?
CAdam/convolutional_autoencoder/convolution_decoder/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*T
shared_nameECAdam/convolutional_autoencoder/convolution_decoder/dense_1/kernel/m
?
WAdam/convolutional_autoencoder/convolution_decoder/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpCAdam/convolutional_autoencoder/convolution_decoder/dense_1/kernel/m*
_output_shapes
:	?*
dtype0
?
AAdam/convolutional_autoencoder/convolution_decoder/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*R
shared_nameCAAdam/convolutional_autoencoder/convolution_decoder/dense_1/bias/m
?
UAdam/convolutional_autoencoder/convolution_decoder/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAAdam/convolutional_autoencoder/convolution_decoder/dense_1/bias/m*
_output_shapes	
:?*
dtype0
?
DAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*U
shared_nameFDAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/kernel/m
?
XAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpDAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/kernel/m*'
_output_shapes
:?*
dtype0
?
BAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*S
shared_nameDBAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/bias/m
?
VAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpBAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/bias/m*
_output_shapes	
:?*
dtype0
?
DAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*U
shared_nameFDAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/kernel/m
?
XAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpDAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/kernel/m*(
_output_shapes
:??*
dtype0
?
BAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*S
shared_nameDBAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/bias/m
?
VAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpBAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/bias/m*
_output_shapes	
:?*
dtype0
?
DAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*U
shared_nameFDAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/kernel/m
?
XAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpDAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/kernel/m*'
_output_shapes
:?@*
dtype0
?
BAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*S
shared_nameDBAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/bias/m
?
VAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpBAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/bias/m*
_output_shapes
:@*
dtype0
?
DAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *U
shared_nameFDAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/kernel/m
?
XAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/kernel/m/Read/ReadVariableOpReadVariableOpDAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/kernel/m*&
_output_shapes
:@ *
dtype0
?
BAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *S
shared_nameDBAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/bias/m
?
VAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/bias/m/Read/ReadVariableOpReadVariableOpBAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/bias/m*
_output_shapes
: *
dtype0
?
DAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *U
shared_nameFDAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/kernel/m
?
XAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/kernel/m/Read/ReadVariableOpReadVariableOpDAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/kernel/m*&
_output_shapes
: *
dtype0
?
BAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*S
shared_nameDBAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/bias/m
?
VAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/bias/m/Read/ReadVariableOpReadVariableOpBAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/bias/m*
_output_shapes
:*
dtype0
?
BAdam/convolutional_autoencoder/convolution_encoder/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*S
shared_nameDBAdam/convolutional_autoencoder/convolution_encoder/conv2d/kernel/v
?
VAdam/convolutional_autoencoder/convolution_encoder/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpBAdam/convolutional_autoencoder/convolution_encoder/conv2d/kernel/v*&
_output_shapes
:*
dtype0
?
@Adam/convolutional_autoencoder/convolution_encoder/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@Adam/convolutional_autoencoder/convolution_encoder/conv2d/bias/v
?
TAdam/convolutional_autoencoder/convolution_encoder/conv2d/bias/v/Read/ReadVariableOpReadVariableOp@Adam/convolutional_autoencoder/convolution_encoder/conv2d/bias/v*
_output_shapes
:*
dtype0
?
DAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *U
shared_nameFDAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/kernel/v
?
XAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpDAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/kernel/v*&
_output_shapes
: *
dtype0
?
BAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *S
shared_nameDBAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/bias/v
?
VAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpBAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/bias/v*
_output_shapes
: *
dtype0
?
DAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*U
shared_nameFDAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/kernel/v
?
XAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpDAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/kernel/v*&
_output_shapes
: @*
dtype0
?
BAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*S
shared_nameDBAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/bias/v
?
VAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpBAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/bias/v*
_output_shapes
:@*
dtype0
?
DAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*U
shared_nameFDAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/kernel/v
?
XAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpDAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/kernel/v*'
_output_shapes
:@?*
dtype0
?
BAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*S
shared_nameDBAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/bias/v
?
VAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpBAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/bias/v*
_output_shapes	
:?*
dtype0
?
AAdam/convolutional_autoencoder/convolution_encoder/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	*R
shared_nameCAAdam/convolutional_autoencoder/convolution_encoder/dense/kernel/v
?
UAdam/convolutional_autoencoder/convolution_encoder/dense/kernel/v/Read/ReadVariableOpReadVariableOpAAdam/convolutional_autoencoder/convolution_encoder/dense/kernel/v*
_output_shapes
:	?	*
dtype0
?
?Adam/convolutional_autoencoder/convolution_encoder/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/convolutional_autoencoder/convolution_encoder/dense/bias/v
?
SAdam/convolutional_autoencoder/convolution_encoder/dense/bias/v/Read/ReadVariableOpReadVariableOp?Adam/convolutional_autoencoder/convolution_encoder/dense/bias/v*
_output_shapes
:*
dtype0
?
CAdam/convolutional_autoencoder/convolution_decoder/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*T
shared_nameECAdam/convolutional_autoencoder/convolution_decoder/dense_1/kernel/v
?
WAdam/convolutional_autoencoder/convolution_decoder/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpCAdam/convolutional_autoencoder/convolution_decoder/dense_1/kernel/v*
_output_shapes
:	?*
dtype0
?
AAdam/convolutional_autoencoder/convolution_decoder/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*R
shared_nameCAAdam/convolutional_autoencoder/convolution_decoder/dense_1/bias/v
?
UAdam/convolutional_autoencoder/convolution_decoder/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAAdam/convolutional_autoencoder/convolution_decoder/dense_1/bias/v*
_output_shapes	
:?*
dtype0
?
DAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*U
shared_nameFDAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/kernel/v
?
XAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpDAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/kernel/v*'
_output_shapes
:?*
dtype0
?
BAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*S
shared_nameDBAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/bias/v
?
VAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpBAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/bias/v*
_output_shapes	
:?*
dtype0
?
DAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*U
shared_nameFDAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/kernel/v
?
XAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpDAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/kernel/v*(
_output_shapes
:??*
dtype0
?
BAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*S
shared_nameDBAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/bias/v
?
VAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpBAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/bias/v*
_output_shapes	
:?*
dtype0
?
DAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*U
shared_nameFDAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/kernel/v
?
XAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpDAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/kernel/v*'
_output_shapes
:?@*
dtype0
?
BAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*S
shared_nameDBAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/bias/v
?
VAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpBAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/bias/v*
_output_shapes
:@*
dtype0
?
DAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *U
shared_nameFDAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/kernel/v
?
XAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/kernel/v/Read/ReadVariableOpReadVariableOpDAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/kernel/v*&
_output_shapes
:@ *
dtype0
?
BAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *S
shared_nameDBAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/bias/v
?
VAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/bias/v/Read/ReadVariableOpReadVariableOpBAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/bias/v*
_output_shapes
: *
dtype0
?
DAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *U
shared_nameFDAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/kernel/v
?
XAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/kernel/v/Read/ReadVariableOpReadVariableOpDAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/kernel/v*&
_output_shapes
: *
dtype0
?
BAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*S
shared_nameDBAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/bias/v
?
VAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/bias/v/Read/ReadVariableOpReadVariableOpBAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??Bޒ B֒
?
	optimizer
encoder
decoder
regularization_losses
trainable_variables
	variables
	keras_api

signatures
?
	iter


beta_1

beta_2
	decay
learning_ratem?m? m?!m?"m?#m?$m?%m?&m?'m?(m?)m?*m?+m?,m?-m?.m?/m?0m?1m?2m?3m?v?v? v?!v?"v?#v?$v?%v?&v?'v?(v?)v?*v?+v?,v?-v?.v?/v?0v?1v?2v?3v?
t
	convs
flat
dense_1
regularization_losses
trainable_variables
	variables
	keras_api
?
dense_1
reshape
	convs
	upsamples

final_conv
regularization_losses
trainable_variables
	variables
	keras_api
 
?
0
1
 2
!3
"4
#5
$6
%7
&8
'9
(10
)11
*12
+13
,14
-15
.16
/17
018
119
220
321
?
0
1
 2
!3
"4
#5
$6
%7
&8
'9
(10
)11
*12
+13
,14
-15
.16
/17
018
119
220
321
?

4layers
regularization_losses
trainable_variables
5layer_metrics
6non_trainable_variables
7metrics
8layer_regularization_losses
	variables
 
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

90
:1
;2
<3
R
=regularization_losses
>trainable_variables
?	variables
@	keras_api
h

&kernel
'bias
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
 
F
0
1
 2
!3
"4
#5
$6
%7
&8
'9
F
0
1
 2
!3
"4
#5
$6
%7
&8
'9
?

Elayers
regularization_losses
trainable_variables
Flayer_metrics
Gnon_trainable_variables
Hmetrics
Ilayer_regularization_losses
	variables
h

(kernel
)bias
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
R
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api

R0
S1
T2
U3

V0
W1
X2
Y3
h

2kernel
3bias
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
 
V
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311
V
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311
?

^layers
regularization_losses
trainable_variables
_layer_metrics
`non_trainable_variables
ametrics
blayer_regularization_losses
	variables
?
VARIABLE_VALUE;convolutional_autoencoder/convolution_encoder/conv2d/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE9convolutional_autoencoder/convolution_encoder/conv2d/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE=convolutional_autoencoder/convolution_encoder/conv2d_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE;convolutional_autoencoder/convolution_encoder/conv2d_1/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE=convolutional_autoencoder/convolution_encoder/conv2d_2/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE;convolutional_autoencoder/convolution_encoder/conv2d_2/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE=convolutional_autoencoder/convolution_encoder/conv2d_3/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE;convolutional_autoencoder/convolution_encoder/conv2d_3/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE:convolutional_autoencoder/convolution_encoder/dense/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE8convolutional_autoencoder/convolution_encoder/dense/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE<convolutional_autoencoder/convolution_decoder/dense_1/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE:convolutional_autoencoder/convolution_decoder/dense_1/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE=convolutional_autoencoder/convolution_decoder/conv2d_4/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE;convolutional_autoencoder/convolution_decoder/conv2d_4/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE=convolutional_autoencoder/convolution_decoder/conv2d_5/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE;convolutional_autoencoder/convolution_decoder/conv2d_5/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE=convolutional_autoencoder/convolution_decoder/conv2d_6/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE;convolutional_autoencoder/convolution_decoder/conv2d_6/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE=convolutional_autoencoder/convolution_decoder/conv2d_7/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE;convolutional_autoencoder/convolution_decoder/conv2d_7/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE=convolutional_autoencoder/convolution_decoder/conv2d_8/kernel1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE;convolutional_autoencoder/convolution_decoder/conv2d_8/bias1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
 

c0
 
h

kernel
bias
dregularization_losses
etrainable_variables
f	variables
g	keras_api
h

 kernel
!bias
hregularization_losses
itrainable_variables
j	variables
k	keras_api
h

"kernel
#bias
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
h

$kernel
%bias
pregularization_losses
qtrainable_variables
r	variables
s	keras_api
 
 
 
?

tlayers
=regularization_losses
>trainable_variables
ulayer_metrics
vnon_trainable_variables
wmetrics
xlayer_regularization_losses
?	variables
 

&0
'1

&0
'1
?

ylayers
Aregularization_losses
Btrainable_variables
zlayer_metrics
{non_trainable_variables
|metrics
}layer_regularization_losses
C	variables
*
90
:1
;2
<3
4
5
 
 
 
 
 

(0
)1

(0
)1
?

~layers
Jregularization_losses
Ktrainable_variables
layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
L	variables
 
 
 
?
?layers
Nregularization_losses
Otrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
P	variables
l

*kernel
+bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

,kernel
-bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

.kernel
/bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

0kernel
1bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 

20
31

20
31
?
?layers
Zregularization_losses
[trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
\	variables
N
0
1
R2
S3
T4
U5
V6
W7
X8
Y9
10
 
 
 
 
8

?total

?count
?	variables
?	keras_api
 

0
1

0
1
?
?layers
dregularization_losses
etrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
f	variables
 

 0
!1

 0
!1
?
?layers
hregularization_losses
itrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
j	variables
 

"0
#1

"0
#1
?
?layers
lregularization_losses
mtrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
n	variables
 

$0
%1

$0
%1
?
?layers
pregularization_losses
qtrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
r	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

*0
+1

*0
+1
?
?layers
?regularization_losses
?trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
 

,0
-1

,0
-1
?
?layers
?regularization_losses
?trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
 

.0
/1

.0
/1
?
?layers
?regularization_losses
?trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
 

00
11

00
11
?
?layers
?regularization_losses
?trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
 
 
 
?
?layers
?regularization_losses
?trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
 
 
 
?
?layers
?regularization_losses
?trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
 
 
 
?
?layers
?regularization_losses
?trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
 
 
 
?
?layers
?regularization_losses
?trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
??
VARIABLE_VALUEBAdam/convolutional_autoencoder/convolution_encoder/conv2d/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@Adam/convolutional_autoencoder/convolution_encoder/conv2d/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAAdam/convolutional_autoencoder/convolution_encoder/dense/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?Adam/convolutional_autoencoder/convolution_encoder/dense/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUECAdam/convolutional_autoencoder/convolution_decoder/dense_1/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAAdam/convolutional_autoencoder/convolution_decoder/dense_1/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/kernel/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/bias/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/kernel/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/bias/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/kernel/mMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/bias/mMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/kernel/mMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/bias/mMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBAdam/convolutional_autoencoder/convolution_encoder/conv2d/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@Adam/convolutional_autoencoder/convolution_encoder/conv2d/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAAdam/convolutional_autoencoder/convolution_encoder/dense/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?Adam/convolutional_autoencoder/convolution_encoder/dense/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUECAdam/convolutional_autoencoder/convolution_decoder/dense_1/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAAdam/convolutional_autoencoder/convolution_decoder/dense_1/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/kernel/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/bias/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/kernel/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/bias/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/kernel/vMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/bias/vMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/kernel/vMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/bias/vMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????@@*
dtype0*$
shape:?????????@@
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1;convolutional_autoencoder/convolution_encoder/conv2d/kernel9convolutional_autoencoder/convolution_encoder/conv2d/bias=convolutional_autoencoder/convolution_encoder/conv2d_1/kernel;convolutional_autoencoder/convolution_encoder/conv2d_1/bias=convolutional_autoencoder/convolution_encoder/conv2d_2/kernel;convolutional_autoencoder/convolution_encoder/conv2d_2/bias=convolutional_autoencoder/convolution_encoder/conv2d_3/kernel;convolutional_autoencoder/convolution_encoder/conv2d_3/bias:convolutional_autoencoder/convolution_encoder/dense/kernel8convolutional_autoencoder/convolution_encoder/dense/bias<convolutional_autoencoder/convolution_decoder/dense_1/kernel:convolutional_autoencoder/convolution_decoder/dense_1/bias=convolutional_autoencoder/convolution_decoder/conv2d_4/kernel;convolutional_autoencoder/convolution_decoder/conv2d_4/bias=convolutional_autoencoder/convolution_decoder/conv2d_5/kernel;convolutional_autoencoder/convolution_decoder/conv2d_5/bias=convolutional_autoencoder/convolution_decoder/conv2d_6/kernel;convolutional_autoencoder/convolution_decoder/conv2d_6/bias=convolutional_autoencoder/convolution_decoder/conv2d_7/kernel;convolutional_autoencoder/convolution_decoder/conv2d_7/bias=convolutional_autoencoder/convolution_decoder/conv2d_8/kernel;convolutional_autoencoder/convolution_decoder/conv2d_8/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_10951
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?0
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpOconvolutional_autoencoder/convolution_encoder/conv2d/kernel/Read/ReadVariableOpMconvolutional_autoencoder/convolution_encoder/conv2d/bias/Read/ReadVariableOpQconvolutional_autoencoder/convolution_encoder/conv2d_1/kernel/Read/ReadVariableOpOconvolutional_autoencoder/convolution_encoder/conv2d_1/bias/Read/ReadVariableOpQconvolutional_autoencoder/convolution_encoder/conv2d_2/kernel/Read/ReadVariableOpOconvolutional_autoencoder/convolution_encoder/conv2d_2/bias/Read/ReadVariableOpQconvolutional_autoencoder/convolution_encoder/conv2d_3/kernel/Read/ReadVariableOpOconvolutional_autoencoder/convolution_encoder/conv2d_3/bias/Read/ReadVariableOpNconvolutional_autoencoder/convolution_encoder/dense/kernel/Read/ReadVariableOpLconvolutional_autoencoder/convolution_encoder/dense/bias/Read/ReadVariableOpPconvolutional_autoencoder/convolution_decoder/dense_1/kernel/Read/ReadVariableOpNconvolutional_autoencoder/convolution_decoder/dense_1/bias/Read/ReadVariableOpQconvolutional_autoencoder/convolution_decoder/conv2d_4/kernel/Read/ReadVariableOpOconvolutional_autoencoder/convolution_decoder/conv2d_4/bias/Read/ReadVariableOpQconvolutional_autoencoder/convolution_decoder/conv2d_5/kernel/Read/ReadVariableOpOconvolutional_autoencoder/convolution_decoder/conv2d_5/bias/Read/ReadVariableOpQconvolutional_autoencoder/convolution_decoder/conv2d_6/kernel/Read/ReadVariableOpOconvolutional_autoencoder/convolution_decoder/conv2d_6/bias/Read/ReadVariableOpQconvolutional_autoencoder/convolution_decoder/conv2d_7/kernel/Read/ReadVariableOpOconvolutional_autoencoder/convolution_decoder/conv2d_7/bias/Read/ReadVariableOpQconvolutional_autoencoder/convolution_decoder/conv2d_8/kernel/Read/ReadVariableOpOconvolutional_autoencoder/convolution_decoder/conv2d_8/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpVAdam/convolutional_autoencoder/convolution_encoder/conv2d/kernel/m/Read/ReadVariableOpTAdam/convolutional_autoencoder/convolution_encoder/conv2d/bias/m/Read/ReadVariableOpXAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/kernel/m/Read/ReadVariableOpVAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/bias/m/Read/ReadVariableOpXAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/kernel/m/Read/ReadVariableOpVAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/bias/m/Read/ReadVariableOpXAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/kernel/m/Read/ReadVariableOpVAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/bias/m/Read/ReadVariableOpUAdam/convolutional_autoencoder/convolution_encoder/dense/kernel/m/Read/ReadVariableOpSAdam/convolutional_autoencoder/convolution_encoder/dense/bias/m/Read/ReadVariableOpWAdam/convolutional_autoencoder/convolution_decoder/dense_1/kernel/m/Read/ReadVariableOpUAdam/convolutional_autoencoder/convolution_decoder/dense_1/bias/m/Read/ReadVariableOpXAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/kernel/m/Read/ReadVariableOpVAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/bias/m/Read/ReadVariableOpXAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/kernel/m/Read/ReadVariableOpVAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/bias/m/Read/ReadVariableOpXAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/kernel/m/Read/ReadVariableOpVAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/bias/m/Read/ReadVariableOpXAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/kernel/m/Read/ReadVariableOpVAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/bias/m/Read/ReadVariableOpXAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/kernel/m/Read/ReadVariableOpVAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/bias/m/Read/ReadVariableOpVAdam/convolutional_autoencoder/convolution_encoder/conv2d/kernel/v/Read/ReadVariableOpTAdam/convolutional_autoencoder/convolution_encoder/conv2d/bias/v/Read/ReadVariableOpXAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/kernel/v/Read/ReadVariableOpVAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/bias/v/Read/ReadVariableOpXAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/kernel/v/Read/ReadVariableOpVAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/bias/v/Read/ReadVariableOpXAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/kernel/v/Read/ReadVariableOpVAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/bias/v/Read/ReadVariableOpUAdam/convolutional_autoencoder/convolution_encoder/dense/kernel/v/Read/ReadVariableOpSAdam/convolutional_autoencoder/convolution_encoder/dense/bias/v/Read/ReadVariableOpWAdam/convolutional_autoencoder/convolution_decoder/dense_1/kernel/v/Read/ReadVariableOpUAdam/convolutional_autoencoder/convolution_decoder/dense_1/bias/v/Read/ReadVariableOpXAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/kernel/v/Read/ReadVariableOpVAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/bias/v/Read/ReadVariableOpXAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/kernel/v/Read/ReadVariableOpVAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/bias/v/Read/ReadVariableOpXAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/kernel/v/Read/ReadVariableOpVAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/bias/v/Read/ReadVariableOpXAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/kernel/v/Read/ReadVariableOpVAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/bias/v/Read/ReadVariableOpXAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/kernel/v/Read/ReadVariableOpVAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/bias/v/Read/ReadVariableOpConst*V
TinO
M2K	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_11440
?%
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate;convolutional_autoencoder/convolution_encoder/conv2d/kernel9convolutional_autoencoder/convolution_encoder/conv2d/bias=convolutional_autoencoder/convolution_encoder/conv2d_1/kernel;convolutional_autoencoder/convolution_encoder/conv2d_1/bias=convolutional_autoencoder/convolution_encoder/conv2d_2/kernel;convolutional_autoencoder/convolution_encoder/conv2d_2/bias=convolutional_autoencoder/convolution_encoder/conv2d_3/kernel;convolutional_autoencoder/convolution_encoder/conv2d_3/bias:convolutional_autoencoder/convolution_encoder/dense/kernel8convolutional_autoencoder/convolution_encoder/dense/bias<convolutional_autoencoder/convolution_decoder/dense_1/kernel:convolutional_autoencoder/convolution_decoder/dense_1/bias=convolutional_autoencoder/convolution_decoder/conv2d_4/kernel;convolutional_autoencoder/convolution_decoder/conv2d_4/bias=convolutional_autoencoder/convolution_decoder/conv2d_5/kernel;convolutional_autoencoder/convolution_decoder/conv2d_5/bias=convolutional_autoencoder/convolution_decoder/conv2d_6/kernel;convolutional_autoencoder/convolution_decoder/conv2d_6/bias=convolutional_autoencoder/convolution_decoder/conv2d_7/kernel;convolutional_autoencoder/convolution_decoder/conv2d_7/bias=convolutional_autoencoder/convolution_decoder/conv2d_8/kernel;convolutional_autoencoder/convolution_decoder/conv2d_8/biastotalcountBAdam/convolutional_autoencoder/convolution_encoder/conv2d/kernel/m@Adam/convolutional_autoencoder/convolution_encoder/conv2d/bias/mDAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/kernel/mBAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/bias/mDAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/kernel/mBAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/bias/mDAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/kernel/mBAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/bias/mAAdam/convolutional_autoencoder/convolution_encoder/dense/kernel/m?Adam/convolutional_autoencoder/convolution_encoder/dense/bias/mCAdam/convolutional_autoencoder/convolution_decoder/dense_1/kernel/mAAdam/convolutional_autoencoder/convolution_decoder/dense_1/bias/mDAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/kernel/mBAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/bias/mDAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/kernel/mBAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/bias/mDAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/kernel/mBAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/bias/mDAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/kernel/mBAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/bias/mDAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/kernel/mBAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/bias/mBAdam/convolutional_autoencoder/convolution_encoder/conv2d/kernel/v@Adam/convolutional_autoencoder/convolution_encoder/conv2d/bias/vDAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/kernel/vBAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/bias/vDAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/kernel/vBAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/bias/vDAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/kernel/vBAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/bias/vAAdam/convolutional_autoencoder/convolution_encoder/dense/kernel/v?Adam/convolutional_autoencoder/convolution_encoder/dense/bias/vCAdam/convolutional_autoencoder/convolution_decoder/dense_1/kernel/vAAdam/convolutional_autoencoder/convolution_decoder/dense_1/bias/vDAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/kernel/vBAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/bias/vDAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/kernel/vBAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/bias/vDAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/kernel/vBAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/bias/vDAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/kernel/vBAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/bias/vDAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/kernel/vBAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/bias/v*U
TinN
L2J*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_11669??
?
|
'__inference_dense_1_layer_call_fn_11000

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_105842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
}
(__inference_conv2d_7_layer_call_fn_11198

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_107172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
N__inference_convolution_encoder_layer_call_and_return_conditional_losses_10468
input_1
conv2d_10341
conv2d_10343
conv2d_1_10368
conv2d_1_10370
conv2d_2_10395
conv2d_2_10397
conv2d_3_10422
conv2d_3_10424
dense_10462
dense_10464
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_10341conv2d_10343*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_103302 
conv2d/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_10368conv2d_1_10370*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_103572"
 conv2d_1/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_10395conv2d_2_10397*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_103842"
 conv2d_2/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_10422conv2d_3_10424*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_104112"
 conv2d_3/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_104332
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_10462dense_10464*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_104512
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????@@::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?
}
(__inference_conv2d_4_layer_call_fn_11138

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_106332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_10507

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
K
/__inference_up_sampling2d_1_layer_call_fn_10532

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_105262
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_1_layer_call_and_return_conditional_losses_10584

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_conv2d_8_layer_call_and_return_conditional_losses_10744

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?=
!__inference__traced_restore_11669
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rateR
Nassignvariableop_5_convolutional_autoencoder_convolution_encoder_conv2d_kernelP
Lassignvariableop_6_convolutional_autoencoder_convolution_encoder_conv2d_biasT
Passignvariableop_7_convolutional_autoencoder_convolution_encoder_conv2d_1_kernelR
Nassignvariableop_8_convolutional_autoencoder_convolution_encoder_conv2d_1_biasT
Passignvariableop_9_convolutional_autoencoder_convolution_encoder_conv2d_2_kernelS
Oassignvariableop_10_convolutional_autoencoder_convolution_encoder_conv2d_2_biasU
Qassignvariableop_11_convolutional_autoencoder_convolution_encoder_conv2d_3_kernelS
Oassignvariableop_12_convolutional_autoencoder_convolution_encoder_conv2d_3_biasR
Nassignvariableop_13_convolutional_autoencoder_convolution_encoder_dense_kernelP
Lassignvariableop_14_convolutional_autoencoder_convolution_encoder_dense_biasT
Passignvariableop_15_convolutional_autoencoder_convolution_decoder_dense_1_kernelR
Nassignvariableop_16_convolutional_autoencoder_convolution_decoder_dense_1_biasU
Qassignvariableop_17_convolutional_autoencoder_convolution_decoder_conv2d_4_kernelS
Oassignvariableop_18_convolutional_autoencoder_convolution_decoder_conv2d_4_biasU
Qassignvariableop_19_convolutional_autoencoder_convolution_decoder_conv2d_5_kernelS
Oassignvariableop_20_convolutional_autoencoder_convolution_decoder_conv2d_5_biasU
Qassignvariableop_21_convolutional_autoencoder_convolution_decoder_conv2d_6_kernelS
Oassignvariableop_22_convolutional_autoencoder_convolution_decoder_conv2d_6_biasU
Qassignvariableop_23_convolutional_autoencoder_convolution_decoder_conv2d_7_kernelS
Oassignvariableop_24_convolutional_autoencoder_convolution_decoder_conv2d_7_biasU
Qassignvariableop_25_convolutional_autoencoder_convolution_decoder_conv2d_8_kernelS
Oassignvariableop_26_convolutional_autoencoder_convolution_decoder_conv2d_8_bias
assignvariableop_27_total
assignvariableop_28_countZ
Vassignvariableop_29_adam_convolutional_autoencoder_convolution_encoder_conv2d_kernel_mX
Tassignvariableop_30_adam_convolutional_autoencoder_convolution_encoder_conv2d_bias_m\
Xassignvariableop_31_adam_convolutional_autoencoder_convolution_encoder_conv2d_1_kernel_mZ
Vassignvariableop_32_adam_convolutional_autoencoder_convolution_encoder_conv2d_1_bias_m\
Xassignvariableop_33_adam_convolutional_autoencoder_convolution_encoder_conv2d_2_kernel_mZ
Vassignvariableop_34_adam_convolutional_autoencoder_convolution_encoder_conv2d_2_bias_m\
Xassignvariableop_35_adam_convolutional_autoencoder_convolution_encoder_conv2d_3_kernel_mZ
Vassignvariableop_36_adam_convolutional_autoencoder_convolution_encoder_conv2d_3_bias_mY
Uassignvariableop_37_adam_convolutional_autoencoder_convolution_encoder_dense_kernel_mW
Sassignvariableop_38_adam_convolutional_autoencoder_convolution_encoder_dense_bias_m[
Wassignvariableop_39_adam_convolutional_autoencoder_convolution_decoder_dense_1_kernel_mY
Uassignvariableop_40_adam_convolutional_autoencoder_convolution_decoder_dense_1_bias_m\
Xassignvariableop_41_adam_convolutional_autoencoder_convolution_decoder_conv2d_4_kernel_mZ
Vassignvariableop_42_adam_convolutional_autoencoder_convolution_decoder_conv2d_4_bias_m\
Xassignvariableop_43_adam_convolutional_autoencoder_convolution_decoder_conv2d_5_kernel_mZ
Vassignvariableop_44_adam_convolutional_autoencoder_convolution_decoder_conv2d_5_bias_m\
Xassignvariableop_45_adam_convolutional_autoencoder_convolution_decoder_conv2d_6_kernel_mZ
Vassignvariableop_46_adam_convolutional_autoencoder_convolution_decoder_conv2d_6_bias_m\
Xassignvariableop_47_adam_convolutional_autoencoder_convolution_decoder_conv2d_7_kernel_mZ
Vassignvariableop_48_adam_convolutional_autoencoder_convolution_decoder_conv2d_7_bias_m\
Xassignvariableop_49_adam_convolutional_autoencoder_convolution_decoder_conv2d_8_kernel_mZ
Vassignvariableop_50_adam_convolutional_autoencoder_convolution_decoder_conv2d_8_bias_mZ
Vassignvariableop_51_adam_convolutional_autoencoder_convolution_encoder_conv2d_kernel_vX
Tassignvariableop_52_adam_convolutional_autoencoder_convolution_encoder_conv2d_bias_v\
Xassignvariableop_53_adam_convolutional_autoencoder_convolution_encoder_conv2d_1_kernel_vZ
Vassignvariableop_54_adam_convolutional_autoencoder_convolution_encoder_conv2d_1_bias_v\
Xassignvariableop_55_adam_convolutional_autoencoder_convolution_encoder_conv2d_2_kernel_vZ
Vassignvariableop_56_adam_convolutional_autoencoder_convolution_encoder_conv2d_2_bias_v\
Xassignvariableop_57_adam_convolutional_autoencoder_convolution_encoder_conv2d_3_kernel_vZ
Vassignvariableop_58_adam_convolutional_autoencoder_convolution_encoder_conv2d_3_bias_vY
Uassignvariableop_59_adam_convolutional_autoencoder_convolution_encoder_dense_kernel_vW
Sassignvariableop_60_adam_convolutional_autoencoder_convolution_encoder_dense_bias_v[
Wassignvariableop_61_adam_convolutional_autoencoder_convolution_decoder_dense_1_kernel_vY
Uassignvariableop_62_adam_convolutional_autoencoder_convolution_decoder_dense_1_bias_v\
Xassignvariableop_63_adam_convolutional_autoencoder_convolution_decoder_conv2d_4_kernel_vZ
Vassignvariableop_64_adam_convolutional_autoencoder_convolution_decoder_conv2d_4_bias_v\
Xassignvariableop_65_adam_convolutional_autoencoder_convolution_decoder_conv2d_5_kernel_vZ
Vassignvariableop_66_adam_convolutional_autoencoder_convolution_decoder_conv2d_5_bias_v\
Xassignvariableop_67_adam_convolutional_autoencoder_convolution_decoder_conv2d_6_kernel_vZ
Vassignvariableop_68_adam_convolutional_autoencoder_convolution_decoder_conv2d_6_bias_v\
Xassignvariableop_69_adam_convolutional_autoencoder_convolution_decoder_conv2d_7_kernel_vZ
Vassignvariableop_70_adam_convolutional_autoencoder_convolution_decoder_conv2d_7_bias_v\
Xassignvariableop_71_adam_convolutional_autoencoder_convolution_decoder_conv2d_8_kernel_vZ
Vassignvariableop_72_adam_convolutional_autoencoder_convolution_decoder_conv2d_8_bias_v
identity_74??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_8?AssignVariableOp_9?'
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*?&
value?&B?&JB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*?
value?B?JB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*X
dtypesN
L2J	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpNassignvariableop_5_convolutional_autoencoder_convolution_encoder_conv2d_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpLassignvariableop_6_convolutional_autoencoder_convolution_encoder_conv2d_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpPassignvariableop_7_convolutional_autoencoder_convolution_encoder_conv2d_1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpNassignvariableop_8_convolutional_autoencoder_convolution_encoder_conv2d_1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpPassignvariableop_9_convolutional_autoencoder_convolution_encoder_conv2d_2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpOassignvariableop_10_convolutional_autoencoder_convolution_encoder_conv2d_2_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpQassignvariableop_11_convolutional_autoencoder_convolution_encoder_conv2d_3_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpOassignvariableop_12_convolutional_autoencoder_convolution_encoder_conv2d_3_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpNassignvariableop_13_convolutional_autoencoder_convolution_encoder_dense_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpLassignvariableop_14_convolutional_autoencoder_convolution_encoder_dense_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpPassignvariableop_15_convolutional_autoencoder_convolution_decoder_dense_1_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpNassignvariableop_16_convolutional_autoencoder_convolution_decoder_dense_1_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpQassignvariableop_17_convolutional_autoencoder_convolution_decoder_conv2d_4_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpOassignvariableop_18_convolutional_autoencoder_convolution_decoder_conv2d_4_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpQassignvariableop_19_convolutional_autoencoder_convolution_decoder_conv2d_5_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpOassignvariableop_20_convolutional_autoencoder_convolution_decoder_conv2d_5_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpQassignvariableop_21_convolutional_autoencoder_convolution_decoder_conv2d_6_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpOassignvariableop_22_convolutional_autoencoder_convolution_decoder_conv2d_6_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpQassignvariableop_23_convolutional_autoencoder_convolution_decoder_conv2d_7_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpOassignvariableop_24_convolutional_autoencoder_convolution_decoder_conv2d_7_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpQassignvariableop_25_convolutional_autoencoder_convolution_decoder_conv2d_8_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpOassignvariableop_26_convolutional_autoencoder_convolution_decoder_conv2d_8_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpVassignvariableop_29_adam_convolutional_autoencoder_convolution_encoder_conv2d_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpTassignvariableop_30_adam_convolutional_autoencoder_convolution_encoder_conv2d_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpXassignvariableop_31_adam_convolutional_autoencoder_convolution_encoder_conv2d_1_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpVassignvariableop_32_adam_convolutional_autoencoder_convolution_encoder_conv2d_1_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpXassignvariableop_33_adam_convolutional_autoencoder_convolution_encoder_conv2d_2_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOpVassignvariableop_34_adam_convolutional_autoencoder_convolution_encoder_conv2d_2_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpXassignvariableop_35_adam_convolutional_autoencoder_convolution_encoder_conv2d_3_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpVassignvariableop_36_adam_convolutional_autoencoder_convolution_encoder_conv2d_3_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpUassignvariableop_37_adam_convolutional_autoencoder_convolution_encoder_dense_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpSassignvariableop_38_adam_convolutional_autoencoder_convolution_encoder_dense_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpWassignvariableop_39_adam_convolutional_autoencoder_convolution_decoder_dense_1_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpUassignvariableop_40_adam_convolutional_autoencoder_convolution_decoder_dense_1_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpXassignvariableop_41_adam_convolutional_autoencoder_convolution_decoder_conv2d_4_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpVassignvariableop_42_adam_convolutional_autoencoder_convolution_decoder_conv2d_4_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpXassignvariableop_43_adam_convolutional_autoencoder_convolution_decoder_conv2d_5_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpVassignvariableop_44_adam_convolutional_autoencoder_convolution_decoder_conv2d_5_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpXassignvariableop_45_adam_convolutional_autoencoder_convolution_decoder_conv2d_6_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOpVassignvariableop_46_adam_convolutional_autoencoder_convolution_decoder_conv2d_6_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOpXassignvariableop_47_adam_convolutional_autoencoder_convolution_decoder_conv2d_7_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOpVassignvariableop_48_adam_convolutional_autoencoder_convolution_decoder_conv2d_7_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOpXassignvariableop_49_adam_convolutional_autoencoder_convolution_decoder_conv2d_8_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpVassignvariableop_50_adam_convolutional_autoencoder_convolution_decoder_conv2d_8_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOpVassignvariableop_51_adam_convolutional_autoencoder_convolution_encoder_conv2d_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOpTassignvariableop_52_adam_convolutional_autoencoder_convolution_encoder_conv2d_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOpXassignvariableop_53_adam_convolutional_autoencoder_convolution_encoder_conv2d_1_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOpVassignvariableop_54_adam_convolutional_autoencoder_convolution_encoder_conv2d_1_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOpXassignvariableop_55_adam_convolutional_autoencoder_convolution_encoder_conv2d_2_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOpVassignvariableop_56_adam_convolutional_autoencoder_convolution_encoder_conv2d_2_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOpXassignvariableop_57_adam_convolutional_autoencoder_convolution_encoder_conv2d_3_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOpVassignvariableop_58_adam_convolutional_autoencoder_convolution_encoder_conv2d_3_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOpUassignvariableop_59_adam_convolutional_autoencoder_convolution_encoder_dense_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOpSassignvariableop_60_adam_convolutional_autoencoder_convolution_encoder_dense_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOpWassignvariableop_61_adam_convolutional_autoencoder_convolution_decoder_dense_1_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOpUassignvariableop_62_adam_convolutional_autoencoder_convolution_decoder_dense_1_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOpXassignvariableop_63_adam_convolutional_autoencoder_convolution_decoder_conv2d_4_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOpVassignvariableop_64_adam_convolutional_autoencoder_convolution_decoder_conv2d_4_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOpXassignvariableop_65_adam_convolutional_autoencoder_convolution_decoder_conv2d_5_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOpVassignvariableop_66_adam_convolutional_autoencoder_convolution_decoder_conv2d_5_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOpXassignvariableop_67_adam_convolutional_autoencoder_convolution_decoder_conv2d_6_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOpVassignvariableop_68_adam_convolutional_autoencoder_convolution_decoder_conv2d_6_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOpXassignvariableop_69_adam_convolutional_autoencoder_convolution_decoder_conv2d_7_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOpVassignvariableop_70_adam_convolutional_autoencoder_convolution_decoder_conv2d_7_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOpXassignvariableop_71_adam_convolutional_autoencoder_convolution_decoder_conv2d_8_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOpVassignvariableop_72_adam_convolutional_autoencoder_convolution_decoder_conv2d_8_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_729
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_73Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_73?
Identity_74IdentityIdentity_73:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_74"#
identity_74Identity_74:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
}
(__inference_conv2d_2_layer_call_fn_11098

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_103842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_10951
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_103152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????@@::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?
}
(__inference_conv2d_1_layer_call_fn_11078

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_103572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_1_layer_call_and_return_conditional_losses_10991

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
K
/__inference_up_sampling2d_3_layer_call_fn_10570

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_105642
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
Խ
?5
__inference__traced_save_11440
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopZ
Vsavev2_convolutional_autoencoder_convolution_encoder_conv2d_kernel_read_readvariableopX
Tsavev2_convolutional_autoencoder_convolution_encoder_conv2d_bias_read_readvariableop\
Xsavev2_convolutional_autoencoder_convolution_encoder_conv2d_1_kernel_read_readvariableopZ
Vsavev2_convolutional_autoencoder_convolution_encoder_conv2d_1_bias_read_readvariableop\
Xsavev2_convolutional_autoencoder_convolution_encoder_conv2d_2_kernel_read_readvariableopZ
Vsavev2_convolutional_autoencoder_convolution_encoder_conv2d_2_bias_read_readvariableop\
Xsavev2_convolutional_autoencoder_convolution_encoder_conv2d_3_kernel_read_readvariableopZ
Vsavev2_convolutional_autoencoder_convolution_encoder_conv2d_3_bias_read_readvariableopY
Usavev2_convolutional_autoencoder_convolution_encoder_dense_kernel_read_readvariableopW
Ssavev2_convolutional_autoencoder_convolution_encoder_dense_bias_read_readvariableop[
Wsavev2_convolutional_autoencoder_convolution_decoder_dense_1_kernel_read_readvariableopY
Usavev2_convolutional_autoencoder_convolution_decoder_dense_1_bias_read_readvariableop\
Xsavev2_convolutional_autoencoder_convolution_decoder_conv2d_4_kernel_read_readvariableopZ
Vsavev2_convolutional_autoencoder_convolution_decoder_conv2d_4_bias_read_readvariableop\
Xsavev2_convolutional_autoencoder_convolution_decoder_conv2d_5_kernel_read_readvariableopZ
Vsavev2_convolutional_autoencoder_convolution_decoder_conv2d_5_bias_read_readvariableop\
Xsavev2_convolutional_autoencoder_convolution_decoder_conv2d_6_kernel_read_readvariableopZ
Vsavev2_convolutional_autoencoder_convolution_decoder_conv2d_6_bias_read_readvariableop\
Xsavev2_convolutional_autoencoder_convolution_decoder_conv2d_7_kernel_read_readvariableopZ
Vsavev2_convolutional_autoencoder_convolution_decoder_conv2d_7_bias_read_readvariableop\
Xsavev2_convolutional_autoencoder_convolution_decoder_conv2d_8_kernel_read_readvariableopZ
Vsavev2_convolutional_autoencoder_convolution_decoder_conv2d_8_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopa
]savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_kernel_m_read_readvariableop_
[savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_bias_m_read_readvariableopc
_savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_1_kernel_m_read_readvariableopa
]savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_1_bias_m_read_readvariableopc
_savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_2_kernel_m_read_readvariableopa
]savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_2_bias_m_read_readvariableopc
_savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_3_kernel_m_read_readvariableopa
]savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_3_bias_m_read_readvariableop`
\savev2_adam_convolutional_autoencoder_convolution_encoder_dense_kernel_m_read_readvariableop^
Zsavev2_adam_convolutional_autoencoder_convolution_encoder_dense_bias_m_read_readvariableopb
^savev2_adam_convolutional_autoencoder_convolution_decoder_dense_1_kernel_m_read_readvariableop`
\savev2_adam_convolutional_autoencoder_convolution_decoder_dense_1_bias_m_read_readvariableopc
_savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_4_kernel_m_read_readvariableopa
]savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_4_bias_m_read_readvariableopc
_savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_5_kernel_m_read_readvariableopa
]savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_5_bias_m_read_readvariableopc
_savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_6_kernel_m_read_readvariableopa
]savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_6_bias_m_read_readvariableopc
_savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_7_kernel_m_read_readvariableopa
]savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_7_bias_m_read_readvariableopc
_savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_8_kernel_m_read_readvariableopa
]savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_8_bias_m_read_readvariableopa
]savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_kernel_v_read_readvariableop_
[savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_bias_v_read_readvariableopc
_savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_1_kernel_v_read_readvariableopa
]savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_1_bias_v_read_readvariableopc
_savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_2_kernel_v_read_readvariableopa
]savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_2_bias_v_read_readvariableopc
_savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_3_kernel_v_read_readvariableopa
]savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_3_bias_v_read_readvariableop`
\savev2_adam_convolutional_autoencoder_convolution_encoder_dense_kernel_v_read_readvariableop^
Zsavev2_adam_convolutional_autoencoder_convolution_encoder_dense_bias_v_read_readvariableopb
^savev2_adam_convolutional_autoencoder_convolution_decoder_dense_1_kernel_v_read_readvariableop`
\savev2_adam_convolutional_autoencoder_convolution_decoder_dense_1_bias_v_read_readvariableopc
_savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_4_kernel_v_read_readvariableopa
]savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_4_bias_v_read_readvariableopc
_savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_5_kernel_v_read_readvariableopa
]savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_5_bias_v_read_readvariableopc
_savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_6_kernel_v_read_readvariableopa
]savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_6_bias_v_read_readvariableopc
_savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_7_kernel_v_read_readvariableopa
]savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_7_bias_v_read_readvariableopc
_savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_8_kernel_v_read_readvariableopa
]savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_8_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?'
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*?&
value?&B?&JB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*?
value?B?JB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?4
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopVsavev2_convolutional_autoencoder_convolution_encoder_conv2d_kernel_read_readvariableopTsavev2_convolutional_autoencoder_convolution_encoder_conv2d_bias_read_readvariableopXsavev2_convolutional_autoencoder_convolution_encoder_conv2d_1_kernel_read_readvariableopVsavev2_convolutional_autoencoder_convolution_encoder_conv2d_1_bias_read_readvariableopXsavev2_convolutional_autoencoder_convolution_encoder_conv2d_2_kernel_read_readvariableopVsavev2_convolutional_autoencoder_convolution_encoder_conv2d_2_bias_read_readvariableopXsavev2_convolutional_autoencoder_convolution_encoder_conv2d_3_kernel_read_readvariableopVsavev2_convolutional_autoencoder_convolution_encoder_conv2d_3_bias_read_readvariableopUsavev2_convolutional_autoencoder_convolution_encoder_dense_kernel_read_readvariableopSsavev2_convolutional_autoencoder_convolution_encoder_dense_bias_read_readvariableopWsavev2_convolutional_autoencoder_convolution_decoder_dense_1_kernel_read_readvariableopUsavev2_convolutional_autoencoder_convolution_decoder_dense_1_bias_read_readvariableopXsavev2_convolutional_autoencoder_convolution_decoder_conv2d_4_kernel_read_readvariableopVsavev2_convolutional_autoencoder_convolution_decoder_conv2d_4_bias_read_readvariableopXsavev2_convolutional_autoencoder_convolution_decoder_conv2d_5_kernel_read_readvariableopVsavev2_convolutional_autoencoder_convolution_decoder_conv2d_5_bias_read_readvariableopXsavev2_convolutional_autoencoder_convolution_decoder_conv2d_6_kernel_read_readvariableopVsavev2_convolutional_autoencoder_convolution_decoder_conv2d_6_bias_read_readvariableopXsavev2_convolutional_autoencoder_convolution_decoder_conv2d_7_kernel_read_readvariableopVsavev2_convolutional_autoencoder_convolution_decoder_conv2d_7_bias_read_readvariableopXsavev2_convolutional_autoencoder_convolution_decoder_conv2d_8_kernel_read_readvariableopVsavev2_convolutional_autoencoder_convolution_decoder_conv2d_8_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop]savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_kernel_m_read_readvariableop[savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_bias_m_read_readvariableop_savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_1_kernel_m_read_readvariableop]savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_1_bias_m_read_readvariableop_savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_2_kernel_m_read_readvariableop]savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_2_bias_m_read_readvariableop_savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_3_kernel_m_read_readvariableop]savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_3_bias_m_read_readvariableop\savev2_adam_convolutional_autoencoder_convolution_encoder_dense_kernel_m_read_readvariableopZsavev2_adam_convolutional_autoencoder_convolution_encoder_dense_bias_m_read_readvariableop^savev2_adam_convolutional_autoencoder_convolution_decoder_dense_1_kernel_m_read_readvariableop\savev2_adam_convolutional_autoencoder_convolution_decoder_dense_1_bias_m_read_readvariableop_savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_4_kernel_m_read_readvariableop]savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_4_bias_m_read_readvariableop_savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_5_kernel_m_read_readvariableop]savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_5_bias_m_read_readvariableop_savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_6_kernel_m_read_readvariableop]savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_6_bias_m_read_readvariableop_savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_7_kernel_m_read_readvariableop]savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_7_bias_m_read_readvariableop_savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_8_kernel_m_read_readvariableop]savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_8_bias_m_read_readvariableop]savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_kernel_v_read_readvariableop[savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_bias_v_read_readvariableop_savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_1_kernel_v_read_readvariableop]savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_1_bias_v_read_readvariableop_savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_2_kernel_v_read_readvariableop]savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_2_bias_v_read_readvariableop_savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_3_kernel_v_read_readvariableop]savev2_adam_convolutional_autoencoder_convolution_encoder_conv2d_3_bias_v_read_readvariableop\savev2_adam_convolutional_autoencoder_convolution_encoder_dense_kernel_v_read_readvariableopZsavev2_adam_convolutional_autoencoder_convolution_encoder_dense_bias_v_read_readvariableop^savev2_adam_convolutional_autoencoder_convolution_decoder_dense_1_kernel_v_read_readvariableop\savev2_adam_convolutional_autoencoder_convolution_decoder_dense_1_bias_v_read_readvariableop_savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_4_kernel_v_read_readvariableop]savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_4_bias_v_read_readvariableop_savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_5_kernel_v_read_readvariableop]savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_5_bias_v_read_readvariableop_savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_6_kernel_v_read_readvariableop]savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_6_bias_v_read_readvariableop_savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_7_kernel_v_read_readvariableop]savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_7_bias_v_read_readvariableop_savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_8_kernel_v_read_readvariableop]savev2_adam_convolutional_autoencoder_convolution_decoder_conv2d_8_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *X
dtypesN
L2J	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : ::: : : @:@:@?:?:	?	::	?:?:?:?:??:?:?@:@:@ : : :: : ::: : : @:@:@?:?:	?	::	?:?:?:?:??:?:?@:@:@ : : :::: : : @:@:@?:?:	?	::	?:?:?:?:??:?:?@:@:@ : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 	

_output_shapes
: :,
(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:%!

_output_shapes
:	?	: 

_output_shapes
::%!

_output_shapes
:	?:!

_output_shapes	
:?:-)
'
_output_shapes
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:-)
'
_output_shapes
:?@: 

_output_shapes
:@:,(
&
_output_shapes
:@ : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::, (
&
_output_shapes
: : !

_output_shapes
: :,"(
&
_output_shapes
: @: #

_output_shapes
:@:-$)
'
_output_shapes
:@?:!%

_output_shapes	
:?:%&!

_output_shapes
:	?	: '

_output_shapes
::%(!

_output_shapes
:	?:!)

_output_shapes	
:?:-*)
'
_output_shapes
:?:!+

_output_shapes	
:?:.,*
(
_output_shapes
:??:!-

_output_shapes	
:?:-.)
'
_output_shapes
:?@: /

_output_shapes
:@:,0(
&
_output_shapes
:@ : 1

_output_shapes
: :,2(
&
_output_shapes
: : 3

_output_shapes
::,4(
&
_output_shapes
:: 5

_output_shapes
::,6(
&
_output_shapes
: : 7

_output_shapes
: :,8(
&
_output_shapes
: @: 9

_output_shapes
:@:-:)
'
_output_shapes
:@?:!;

_output_shapes	
:?:%<!

_output_shapes
:	?	: =

_output_shapes
::%>!

_output_shapes
:	?:!?

_output_shapes	
:?:-@)
'
_output_shapes
:?:!A

_output_shapes	
:?:.B*
(
_output_shapes
:??:!C

_output_shapes	
:?:-D)
'
_output_shapes
:?@: E

_output_shapes
:@:,F(
&
_output_shapes
:@ : G

_output_shapes
: :,H(
&
_output_shapes
: : I

_output_shapes
::J

_output_shapes
: 
?
I
-__inference_up_sampling2d_layer_call_fn_10513

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_105072
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
@__inference_dense_layer_call_and_return_conditional_losses_10972

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?

?
A__inference_conv2d_layer_call_and_return_conditional_losses_11049

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_10433

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_11129

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_6_layer_call_and_return_conditional_losses_11169

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
C
'__inference_flatten_layer_call_fn_10962

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_104332
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
A__inference_conv2d_layer_call_and_return_conditional_losses_10330

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?	
?
3__inference_convolution_decoder_layer_call_fn_10791
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_convolution_decoder_layer_call_and_return_conditional_losses_107612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
3__inference_convolution_encoder_layer_call_fn_10494
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_convolution_encoder_layer_call_and_return_conditional_losses_104682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????@@::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?

?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_11109

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
}
(__inference_conv2d_8_layer_call_fn_11038

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_107442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
f
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_10526

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
}
(__inference_conv2d_3_layer_call_fn_11118

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_104112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
^
B__inference_reshape_layer_call_and_return_conditional_losses_10614

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
T__inference_convolutional_autoencoder_layer_call_and_return_conditional_losses_10842
input_1
convolution_encoder_10795
convolution_encoder_10797
convolution_encoder_10799
convolution_encoder_10801
convolution_encoder_10803
convolution_encoder_10805
convolution_encoder_10807
convolution_encoder_10809
convolution_encoder_10811
convolution_encoder_10813
convolution_decoder_10816
convolution_decoder_10818
convolution_decoder_10820
convolution_decoder_10822
convolution_decoder_10824
convolution_decoder_10826
convolution_decoder_10828
convolution_decoder_10830
convolution_decoder_10832
convolution_decoder_10834
convolution_decoder_10836
convolution_decoder_10838
identity??+convolution_decoder/StatefulPartitionedCall?+convolution_encoder/StatefulPartitionedCall?
+convolution_encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1convolution_encoder_10795convolution_encoder_10797convolution_encoder_10799convolution_encoder_10801convolution_encoder_10803convolution_encoder_10805convolution_encoder_10807convolution_encoder_10809convolution_encoder_10811convolution_encoder_10813*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_convolution_encoder_layer_call_and_return_conditional_losses_104682-
+convolution_encoder/StatefulPartitionedCall?
+convolution_decoder/StatefulPartitionedCallStatefulPartitionedCall4convolution_encoder/StatefulPartitionedCall:output:0convolution_decoder_10816convolution_decoder_10818convolution_decoder_10820convolution_decoder_10822convolution_decoder_10824convolution_decoder_10826convolution_decoder_10828convolution_decoder_10830convolution_decoder_10832convolution_decoder_10834convolution_decoder_10836convolution_decoder_10838*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_convolution_decoder_layer_call_and_return_conditional_losses_107612-
+convolution_decoder/StatefulPartitionedCall?
IdentityIdentity4convolution_decoder/StatefulPartitionedCall:output:0,^convolution_decoder/StatefulPartitionedCall,^convolution_encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????@@::::::::::::::::::::::2Z
+convolution_decoder/StatefulPartitionedCall+convolution_decoder/StatefulPartitionedCall2Z
+convolution_encoder/StatefulPartitionedCall+convolution_encoder/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?
}
(__inference_conv2d_6_layer_call_fn_11178

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_106892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_7_layer_call_and_return_conditional_losses_11189

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?

?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_10411

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
@__inference_dense_layer_call_and_return_conditional_losses_10451

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
?
C__inference_conv2d_7_layer_call_and_return_conditional_losses_10717

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
z
%__inference_dense_layer_call_fn_10981

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_104512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
K
/__inference_up_sampling2d_2_layer_call_fn_10551

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_105452
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?1
?
N__inference_convolution_decoder_layer_call_and_return_conditional_losses_10761
input_1
dense_1_10595
dense_1_10597
conv2d_4_10644
conv2d_4_10646
conv2d_5_10672
conv2d_5_10674
conv2d_6_10700
conv2d_6_10702
conv2d_7_10728
conv2d_7_10730
conv2d_8_10755
conv2d_8_10757
identity?? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_1_10595dense_1_10597*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_105842!
dense_1/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_106142
reshape/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_4_10644conv2d_4_10646*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_106332"
 conv2d_4/StatefulPartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_105072
up_sampling2d/PartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_5_10672conv2d_5_10674*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_106612"
 conv2d_5/StatefulPartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_105262!
up_sampling2d_1/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_6_10700conv2d_6_10702*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_106892"
 conv2d_6/StatefulPartitionedCall?
up_sampling2d_2/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_105452!
up_sampling2d_2/PartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_7_10728conv2d_7_10730*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_107172"
 conv2d_7/StatefulPartitionedCall?
up_sampling2d_3/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_105642!
up_sampling2d_3/PartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_8_10755conv2d_8_10757*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_107442"
 conv2d_8/StatefulPartitionedCall?
IdentityIdentity)conv2d_8/StatefulPartitionedCall:output:0!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::::2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_11069

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_reshape_layer_call_and_return_conditional_losses_11014

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_10633

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_10384

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
C__inference_conv2d_8_layer_call_and_return_conditional_losses_11029

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_11149

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
9__inference_convolutional_autoencoder_layer_call_fn_10892
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_convolutional_autoencoder_layer_call_and_return_conditional_losses_108422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????@@::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
??
?
 __inference__wrapped_model_10315
input_1W
Sconvolutional_autoencoder_convolution_encoder_conv2d_conv2d_readvariableop_resourceX
Tconvolutional_autoencoder_convolution_encoder_conv2d_biasadd_readvariableop_resourceY
Uconvolutional_autoencoder_convolution_encoder_conv2d_1_conv2d_readvariableop_resourceZ
Vconvolutional_autoencoder_convolution_encoder_conv2d_1_biasadd_readvariableop_resourceY
Uconvolutional_autoencoder_convolution_encoder_conv2d_2_conv2d_readvariableop_resourceZ
Vconvolutional_autoencoder_convolution_encoder_conv2d_2_biasadd_readvariableop_resourceY
Uconvolutional_autoencoder_convolution_encoder_conv2d_3_conv2d_readvariableop_resourceZ
Vconvolutional_autoencoder_convolution_encoder_conv2d_3_biasadd_readvariableop_resourceV
Rconvolutional_autoencoder_convolution_encoder_dense_matmul_readvariableop_resourceW
Sconvolutional_autoencoder_convolution_encoder_dense_biasadd_readvariableop_resourceX
Tconvolutional_autoencoder_convolution_decoder_dense_1_matmul_readvariableop_resourceY
Uconvolutional_autoencoder_convolution_decoder_dense_1_biasadd_readvariableop_resourceY
Uconvolutional_autoencoder_convolution_decoder_conv2d_4_conv2d_readvariableop_resourceZ
Vconvolutional_autoencoder_convolution_decoder_conv2d_4_biasadd_readvariableop_resourceY
Uconvolutional_autoencoder_convolution_decoder_conv2d_5_conv2d_readvariableop_resourceZ
Vconvolutional_autoencoder_convolution_decoder_conv2d_5_biasadd_readvariableop_resourceY
Uconvolutional_autoencoder_convolution_decoder_conv2d_6_conv2d_readvariableop_resourceZ
Vconvolutional_autoencoder_convolution_decoder_conv2d_6_biasadd_readvariableop_resourceY
Uconvolutional_autoencoder_convolution_decoder_conv2d_7_conv2d_readvariableop_resourceZ
Vconvolutional_autoencoder_convolution_decoder_conv2d_7_biasadd_readvariableop_resourceY
Uconvolutional_autoencoder_convolution_decoder_conv2d_8_conv2d_readvariableop_resourceZ
Vconvolutional_autoencoder_convolution_decoder_conv2d_8_biasadd_readvariableop_resource
identity??Mconvolutional_autoencoder/convolution_decoder/conv2d_4/BiasAdd/ReadVariableOp?Lconvolutional_autoencoder/convolution_decoder/conv2d_4/Conv2D/ReadVariableOp?Mconvolutional_autoencoder/convolution_decoder/conv2d_5/BiasAdd/ReadVariableOp?Lconvolutional_autoencoder/convolution_decoder/conv2d_5/Conv2D/ReadVariableOp?Mconvolutional_autoencoder/convolution_decoder/conv2d_6/BiasAdd/ReadVariableOp?Lconvolutional_autoencoder/convolution_decoder/conv2d_6/Conv2D/ReadVariableOp?Mconvolutional_autoencoder/convolution_decoder/conv2d_7/BiasAdd/ReadVariableOp?Lconvolutional_autoencoder/convolution_decoder/conv2d_7/Conv2D/ReadVariableOp?Mconvolutional_autoencoder/convolution_decoder/conv2d_8/BiasAdd/ReadVariableOp?Lconvolutional_autoencoder/convolution_decoder/conv2d_8/Conv2D/ReadVariableOp?Lconvolutional_autoencoder/convolution_decoder/dense_1/BiasAdd/ReadVariableOp?Kconvolutional_autoencoder/convolution_decoder/dense_1/MatMul/ReadVariableOp?Kconvolutional_autoencoder/convolution_encoder/conv2d/BiasAdd/ReadVariableOp?Jconvolutional_autoencoder/convolution_encoder/conv2d/Conv2D/ReadVariableOp?Mconvolutional_autoencoder/convolution_encoder/conv2d_1/BiasAdd/ReadVariableOp?Lconvolutional_autoencoder/convolution_encoder/conv2d_1/Conv2D/ReadVariableOp?Mconvolutional_autoencoder/convolution_encoder/conv2d_2/BiasAdd/ReadVariableOp?Lconvolutional_autoencoder/convolution_encoder/conv2d_2/Conv2D/ReadVariableOp?Mconvolutional_autoencoder/convolution_encoder/conv2d_3/BiasAdd/ReadVariableOp?Lconvolutional_autoencoder/convolution_encoder/conv2d_3/Conv2D/ReadVariableOp?Jconvolutional_autoencoder/convolution_encoder/dense/BiasAdd/ReadVariableOp?Iconvolutional_autoencoder/convolution_encoder/dense/MatMul/ReadVariableOp?
Jconvolutional_autoencoder/convolution_encoder/conv2d/Conv2D/ReadVariableOpReadVariableOpSconvolutional_autoencoder_convolution_encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02L
Jconvolutional_autoencoder/convolution_encoder/conv2d/Conv2D/ReadVariableOp?
;convolutional_autoencoder/convolution_encoder/conv2d/Conv2DConv2Dinput_1Rconvolutional_autoencoder/convolution_encoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2=
;convolutional_autoencoder/convolution_encoder/conv2d/Conv2D?
Kconvolutional_autoencoder/convolution_encoder/conv2d/BiasAdd/ReadVariableOpReadVariableOpTconvolutional_autoencoder_convolution_encoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02M
Kconvolutional_autoencoder/convolution_encoder/conv2d/BiasAdd/ReadVariableOp?
<convolutional_autoencoder/convolution_encoder/conv2d/BiasAddBiasAddDconvolutional_autoencoder/convolution_encoder/conv2d/Conv2D:output:0Sconvolutional_autoencoder/convolution_encoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2>
<convolutional_autoencoder/convolution_encoder/conv2d/BiasAdd?
9convolutional_autoencoder/convolution_encoder/conv2d/ReluReluEconvolutional_autoencoder/convolution_encoder/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2;
9convolutional_autoencoder/convolution_encoder/conv2d/Relu?
Lconvolutional_autoencoder/convolution_encoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOpUconvolutional_autoencoder_convolution_encoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02N
Lconvolutional_autoencoder/convolution_encoder/conv2d_1/Conv2D/ReadVariableOp?
=convolutional_autoencoder/convolution_encoder/conv2d_1/Conv2DConv2DGconvolutional_autoencoder/convolution_encoder/conv2d/Relu:activations:0Tconvolutional_autoencoder/convolution_encoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2?
=convolutional_autoencoder/convolution_encoder/conv2d_1/Conv2D?
Mconvolutional_autoencoder/convolution_encoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpVconvolutional_autoencoder_convolution_encoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02O
Mconvolutional_autoencoder/convolution_encoder/conv2d_1/BiasAdd/ReadVariableOp?
>convolutional_autoencoder/convolution_encoder/conv2d_1/BiasAddBiasAddFconvolutional_autoencoder/convolution_encoder/conv2d_1/Conv2D:output:0Uconvolutional_autoencoder/convolution_encoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2@
>convolutional_autoencoder/convolution_encoder/conv2d_1/BiasAdd?
;convolutional_autoencoder/convolution_encoder/conv2d_1/ReluReluGconvolutional_autoencoder/convolution_encoder/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2=
;convolutional_autoencoder/convolution_encoder/conv2d_1/Relu?
Lconvolutional_autoencoder/convolution_encoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOpUconvolutional_autoencoder_convolution_encoder_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02N
Lconvolutional_autoencoder/convolution_encoder/conv2d_2/Conv2D/ReadVariableOp?
=convolutional_autoencoder/convolution_encoder/conv2d_2/Conv2DConv2DIconvolutional_autoencoder/convolution_encoder/conv2d_1/Relu:activations:0Tconvolutional_autoencoder/convolution_encoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2?
=convolutional_autoencoder/convolution_encoder/conv2d_2/Conv2D?
Mconvolutional_autoencoder/convolution_encoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpVconvolutional_autoencoder_convolution_encoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02O
Mconvolutional_autoencoder/convolution_encoder/conv2d_2/BiasAdd/ReadVariableOp?
>convolutional_autoencoder/convolution_encoder/conv2d_2/BiasAddBiasAddFconvolutional_autoencoder/convolution_encoder/conv2d_2/Conv2D:output:0Uconvolutional_autoencoder/convolution_encoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2@
>convolutional_autoencoder/convolution_encoder/conv2d_2/BiasAdd?
;convolutional_autoencoder/convolution_encoder/conv2d_2/ReluReluGconvolutional_autoencoder/convolution_encoder/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2=
;convolutional_autoencoder/convolution_encoder/conv2d_2/Relu?
Lconvolutional_autoencoder/convolution_encoder/conv2d_3/Conv2D/ReadVariableOpReadVariableOpUconvolutional_autoencoder_convolution_encoder_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02N
Lconvolutional_autoencoder/convolution_encoder/conv2d_3/Conv2D/ReadVariableOp?
=convolutional_autoencoder/convolution_encoder/conv2d_3/Conv2DConv2DIconvolutional_autoencoder/convolution_encoder/conv2d_2/Relu:activations:0Tconvolutional_autoencoder/convolution_encoder/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2?
=convolutional_autoencoder/convolution_encoder/conv2d_3/Conv2D?
Mconvolutional_autoencoder/convolution_encoder/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpVconvolutional_autoencoder_convolution_encoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02O
Mconvolutional_autoencoder/convolution_encoder/conv2d_3/BiasAdd/ReadVariableOp?
>convolutional_autoencoder/convolution_encoder/conv2d_3/BiasAddBiasAddFconvolutional_autoencoder/convolution_encoder/conv2d_3/Conv2D:output:0Uconvolutional_autoencoder/convolution_encoder/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2@
>convolutional_autoencoder/convolution_encoder/conv2d_3/BiasAdd?
;convolutional_autoencoder/convolution_encoder/conv2d_3/ReluReluGconvolutional_autoencoder/convolution_encoder/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2=
;convolutional_autoencoder/convolution_encoder/conv2d_3/Relu?
;convolutional_autoencoder/convolution_encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2=
;convolutional_autoencoder/convolution_encoder/flatten/Const?
=convolutional_autoencoder/convolution_encoder/flatten/ReshapeReshapeIconvolutional_autoencoder/convolution_encoder/conv2d_3/Relu:activations:0Dconvolutional_autoencoder/convolution_encoder/flatten/Const:output:0*
T0*(
_output_shapes
:??????????	2?
=convolutional_autoencoder/convolution_encoder/flatten/Reshape?
Iconvolutional_autoencoder/convolution_encoder/dense/MatMul/ReadVariableOpReadVariableOpRconvolutional_autoencoder_convolution_encoder_dense_matmul_readvariableop_resource*
_output_shapes
:	?	*
dtype02K
Iconvolutional_autoencoder/convolution_encoder/dense/MatMul/ReadVariableOp?
:convolutional_autoencoder/convolution_encoder/dense/MatMulMatMulFconvolutional_autoencoder/convolution_encoder/flatten/Reshape:output:0Qconvolutional_autoencoder/convolution_encoder/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2<
:convolutional_autoencoder/convolution_encoder/dense/MatMul?
Jconvolutional_autoencoder/convolution_encoder/dense/BiasAdd/ReadVariableOpReadVariableOpSconvolutional_autoencoder_convolution_encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02L
Jconvolutional_autoencoder/convolution_encoder/dense/BiasAdd/ReadVariableOp?
;convolutional_autoencoder/convolution_encoder/dense/BiasAddBiasAddDconvolutional_autoencoder/convolution_encoder/dense/MatMul:product:0Rconvolutional_autoencoder/convolution_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2=
;convolutional_autoencoder/convolution_encoder/dense/BiasAdd?
Kconvolutional_autoencoder/convolution_decoder/dense_1/MatMul/ReadVariableOpReadVariableOpTconvolutional_autoencoder_convolution_decoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02M
Kconvolutional_autoencoder/convolution_decoder/dense_1/MatMul/ReadVariableOp?
<convolutional_autoencoder/convolution_decoder/dense_1/MatMulMatMulDconvolutional_autoencoder/convolution_encoder/dense/BiasAdd:output:0Sconvolutional_autoencoder/convolution_decoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2>
<convolutional_autoencoder/convolution_decoder/dense_1/MatMul?
Lconvolutional_autoencoder/convolution_decoder/dense_1/BiasAdd/ReadVariableOpReadVariableOpUconvolutional_autoencoder_convolution_decoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02N
Lconvolutional_autoencoder/convolution_decoder/dense_1/BiasAdd/ReadVariableOp?
=convolutional_autoencoder/convolution_decoder/dense_1/BiasAddBiasAddFconvolutional_autoencoder/convolution_decoder/dense_1/MatMul:product:0Tconvolutional_autoencoder/convolution_decoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2?
=convolutional_autoencoder/convolution_decoder/dense_1/BiasAdd?
;convolutional_autoencoder/convolution_decoder/reshape/ShapeShapeFconvolutional_autoencoder/convolution_decoder/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:2=
;convolutional_autoencoder/convolution_decoder/reshape/Shape?
Iconvolutional_autoencoder/convolution_decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2K
Iconvolutional_autoencoder/convolution_decoder/reshape/strided_slice/stack?
Kconvolutional_autoencoder/convolution_decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2M
Kconvolutional_autoencoder/convolution_decoder/reshape/strided_slice/stack_1?
Kconvolutional_autoencoder/convolution_decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2M
Kconvolutional_autoencoder/convolution_decoder/reshape/strided_slice/stack_2?
Cconvolutional_autoencoder/convolution_decoder/reshape/strided_sliceStridedSliceDconvolutional_autoencoder/convolution_decoder/reshape/Shape:output:0Rconvolutional_autoencoder/convolution_decoder/reshape/strided_slice/stack:output:0Tconvolutional_autoencoder/convolution_decoder/reshape/strided_slice/stack_1:output:0Tconvolutional_autoencoder/convolution_decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2E
Cconvolutional_autoencoder/convolution_decoder/reshape/strided_slice?
Econvolutional_autoencoder/convolution_decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2G
Econvolutional_autoencoder/convolution_decoder/reshape/Reshape/shape/1?
Econvolutional_autoencoder/convolution_decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2G
Econvolutional_autoencoder/convolution_decoder/reshape/Reshape/shape/2?
Econvolutional_autoencoder/convolution_decoder/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2G
Econvolutional_autoencoder/convolution_decoder/reshape/Reshape/shape/3?
Cconvolutional_autoencoder/convolution_decoder/reshape/Reshape/shapePackLconvolutional_autoencoder/convolution_decoder/reshape/strided_slice:output:0Nconvolutional_autoencoder/convolution_decoder/reshape/Reshape/shape/1:output:0Nconvolutional_autoencoder/convolution_decoder/reshape/Reshape/shape/2:output:0Nconvolutional_autoencoder/convolution_decoder/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2E
Cconvolutional_autoencoder/convolution_decoder/reshape/Reshape/shape?
=convolutional_autoencoder/convolution_decoder/reshape/ReshapeReshapeFconvolutional_autoencoder/convolution_decoder/dense_1/BiasAdd:output:0Lconvolutional_autoencoder/convolution_decoder/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2?
=convolutional_autoencoder/convolution_decoder/reshape/Reshape?
Lconvolutional_autoencoder/convolution_decoder/conv2d_4/Conv2D/ReadVariableOpReadVariableOpUconvolutional_autoencoder_convolution_decoder_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02N
Lconvolutional_autoencoder/convolution_decoder/conv2d_4/Conv2D/ReadVariableOp?
=convolutional_autoencoder/convolution_decoder/conv2d_4/Conv2DConv2DFconvolutional_autoencoder/convolution_decoder/reshape/Reshape:output:0Tconvolutional_autoencoder/convolution_decoder/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2?
=convolutional_autoencoder/convolution_decoder/conv2d_4/Conv2D?
Mconvolutional_autoencoder/convolution_decoder/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpVconvolutional_autoencoder_convolution_decoder_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02O
Mconvolutional_autoencoder/convolution_decoder/conv2d_4/BiasAdd/ReadVariableOp?
>convolutional_autoencoder/convolution_decoder/conv2d_4/BiasAddBiasAddFconvolutional_autoencoder/convolution_decoder/conv2d_4/Conv2D:output:0Uconvolutional_autoencoder/convolution_decoder/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2@
>convolutional_autoencoder/convolution_decoder/conv2d_4/BiasAdd?
;convolutional_autoencoder/convolution_decoder/conv2d_4/ReluReluGconvolutional_autoencoder/convolution_decoder/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2=
;convolutional_autoencoder/convolution_decoder/conv2d_4/Relu?
Aconvolutional_autoencoder/convolution_decoder/up_sampling2d/ShapeShapeIconvolutional_autoencoder/convolution_decoder/conv2d_4/Relu:activations:0*
T0*
_output_shapes
:2C
Aconvolutional_autoencoder/convolution_decoder/up_sampling2d/Shape?
Oconvolutional_autoencoder/convolution_decoder/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2Q
Oconvolutional_autoencoder/convolution_decoder/up_sampling2d/strided_slice/stack?
Qconvolutional_autoencoder/convolution_decoder/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2S
Qconvolutional_autoencoder/convolution_decoder/up_sampling2d/strided_slice/stack_1?
Qconvolutional_autoencoder/convolution_decoder/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
Qconvolutional_autoencoder/convolution_decoder/up_sampling2d/strided_slice/stack_2?
Iconvolutional_autoencoder/convolution_decoder/up_sampling2d/strided_sliceStridedSliceJconvolutional_autoencoder/convolution_decoder/up_sampling2d/Shape:output:0Xconvolutional_autoencoder/convolution_decoder/up_sampling2d/strided_slice/stack:output:0Zconvolutional_autoencoder/convolution_decoder/up_sampling2d/strided_slice/stack_1:output:0Zconvolutional_autoencoder/convolution_decoder/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2K
Iconvolutional_autoencoder/convolution_decoder/up_sampling2d/strided_slice?
Aconvolutional_autoencoder/convolution_decoder/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2C
Aconvolutional_autoencoder/convolution_decoder/up_sampling2d/Const?
?convolutional_autoencoder/convolution_decoder/up_sampling2d/mulMulRconvolutional_autoencoder/convolution_decoder/up_sampling2d/strided_slice:output:0Jconvolutional_autoencoder/convolution_decoder/up_sampling2d/Const:output:0*
T0*
_output_shapes
:2A
?convolutional_autoencoder/convolution_decoder/up_sampling2d/mul?
Xconvolutional_autoencoder/convolution_decoder/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborIconvolutional_autoencoder/convolution_decoder/conv2d_4/Relu:activations:0Cconvolutional_autoencoder/convolution_decoder/up_sampling2d/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2Z
Xconvolutional_autoencoder/convolution_decoder/up_sampling2d/resize/ResizeNearestNeighbor?
Lconvolutional_autoencoder/convolution_decoder/conv2d_5/Conv2D/ReadVariableOpReadVariableOpUconvolutional_autoencoder_convolution_decoder_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02N
Lconvolutional_autoencoder/convolution_decoder/conv2d_5/Conv2D/ReadVariableOp?
=convolutional_autoencoder/convolution_decoder/conv2d_5/Conv2DConv2Diconvolutional_autoencoder/convolution_decoder/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0Tconvolutional_autoencoder/convolution_decoder/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2?
=convolutional_autoencoder/convolution_decoder/conv2d_5/Conv2D?
Mconvolutional_autoencoder/convolution_decoder/conv2d_5/BiasAdd/ReadVariableOpReadVariableOpVconvolutional_autoencoder_convolution_decoder_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02O
Mconvolutional_autoencoder/convolution_decoder/conv2d_5/BiasAdd/ReadVariableOp?
>convolutional_autoencoder/convolution_decoder/conv2d_5/BiasAddBiasAddFconvolutional_autoencoder/convolution_decoder/conv2d_5/Conv2D:output:0Uconvolutional_autoencoder/convolution_decoder/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2@
>convolutional_autoencoder/convolution_decoder/conv2d_5/BiasAdd?
;convolutional_autoencoder/convolution_decoder/conv2d_5/ReluReluGconvolutional_autoencoder/convolution_decoder/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2=
;convolutional_autoencoder/convolution_decoder/conv2d_5/Relu?
Cconvolutional_autoencoder/convolution_decoder/up_sampling2d_1/ShapeShapeIconvolutional_autoencoder/convolution_decoder/conv2d_5/Relu:activations:0*
T0*
_output_shapes
:2E
Cconvolutional_autoencoder/convolution_decoder/up_sampling2d_1/Shape?
Qconvolutional_autoencoder/convolution_decoder/up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2S
Qconvolutional_autoencoder/convolution_decoder/up_sampling2d_1/strided_slice/stack?
Sconvolutional_autoencoder/convolution_decoder/up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2U
Sconvolutional_autoencoder/convolution_decoder/up_sampling2d_1/strided_slice/stack_1?
Sconvolutional_autoencoder/convolution_decoder/up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2U
Sconvolutional_autoencoder/convolution_decoder/up_sampling2d_1/strided_slice/stack_2?
Kconvolutional_autoencoder/convolution_decoder/up_sampling2d_1/strided_sliceStridedSliceLconvolutional_autoencoder/convolution_decoder/up_sampling2d_1/Shape:output:0Zconvolutional_autoencoder/convolution_decoder/up_sampling2d_1/strided_slice/stack:output:0\convolutional_autoencoder/convolution_decoder/up_sampling2d_1/strided_slice/stack_1:output:0\convolutional_autoencoder/convolution_decoder/up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2M
Kconvolutional_autoencoder/convolution_decoder/up_sampling2d_1/strided_slice?
Cconvolutional_autoencoder/convolution_decoder/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2E
Cconvolutional_autoencoder/convolution_decoder/up_sampling2d_1/Const?
Aconvolutional_autoencoder/convolution_decoder/up_sampling2d_1/mulMulTconvolutional_autoencoder/convolution_decoder/up_sampling2d_1/strided_slice:output:0Lconvolutional_autoencoder/convolution_decoder/up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2C
Aconvolutional_autoencoder/convolution_decoder/up_sampling2d_1/mul?
Zconvolutional_autoencoder/convolution_decoder/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighborIconvolutional_autoencoder/convolution_decoder/conv2d_5/Relu:activations:0Econvolutional_autoencoder/convolution_decoder/up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2\
Zconvolutional_autoencoder/convolution_decoder/up_sampling2d_1/resize/ResizeNearestNeighbor?
Lconvolutional_autoencoder/convolution_decoder/conv2d_6/Conv2D/ReadVariableOpReadVariableOpUconvolutional_autoencoder_convolution_decoder_conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02N
Lconvolutional_autoencoder/convolution_decoder/conv2d_6/Conv2D/ReadVariableOp?
=convolutional_autoencoder/convolution_decoder/conv2d_6/Conv2DConv2Dkconvolutional_autoencoder/convolution_decoder/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0Tconvolutional_autoencoder/convolution_decoder/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2?
=convolutional_autoencoder/convolution_decoder/conv2d_6/Conv2D?
Mconvolutional_autoencoder/convolution_decoder/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpVconvolutional_autoencoder_convolution_decoder_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02O
Mconvolutional_autoencoder/convolution_decoder/conv2d_6/BiasAdd/ReadVariableOp?
>convolutional_autoencoder/convolution_decoder/conv2d_6/BiasAddBiasAddFconvolutional_autoencoder/convolution_decoder/conv2d_6/Conv2D:output:0Uconvolutional_autoencoder/convolution_decoder/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2@
>convolutional_autoencoder/convolution_decoder/conv2d_6/BiasAdd?
;convolutional_autoencoder/convolution_decoder/conv2d_6/ReluReluGconvolutional_autoencoder/convolution_decoder/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2=
;convolutional_autoencoder/convolution_decoder/conv2d_6/Relu?
Cconvolutional_autoencoder/convolution_decoder/up_sampling2d_2/ShapeShapeIconvolutional_autoencoder/convolution_decoder/conv2d_6/Relu:activations:0*
T0*
_output_shapes
:2E
Cconvolutional_autoencoder/convolution_decoder/up_sampling2d_2/Shape?
Qconvolutional_autoencoder/convolution_decoder/up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2S
Qconvolutional_autoencoder/convolution_decoder/up_sampling2d_2/strided_slice/stack?
Sconvolutional_autoencoder/convolution_decoder/up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2U
Sconvolutional_autoencoder/convolution_decoder/up_sampling2d_2/strided_slice/stack_1?
Sconvolutional_autoencoder/convolution_decoder/up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2U
Sconvolutional_autoencoder/convolution_decoder/up_sampling2d_2/strided_slice/stack_2?
Kconvolutional_autoencoder/convolution_decoder/up_sampling2d_2/strided_sliceStridedSliceLconvolutional_autoencoder/convolution_decoder/up_sampling2d_2/Shape:output:0Zconvolutional_autoencoder/convolution_decoder/up_sampling2d_2/strided_slice/stack:output:0\convolutional_autoencoder/convolution_decoder/up_sampling2d_2/strided_slice/stack_1:output:0\convolutional_autoencoder/convolution_decoder/up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2M
Kconvolutional_autoencoder/convolution_decoder/up_sampling2d_2/strided_slice?
Cconvolutional_autoencoder/convolution_decoder/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2E
Cconvolutional_autoencoder/convolution_decoder/up_sampling2d_2/Const?
Aconvolutional_autoencoder/convolution_decoder/up_sampling2d_2/mulMulTconvolutional_autoencoder/convolution_decoder/up_sampling2d_2/strided_slice:output:0Lconvolutional_autoencoder/convolution_decoder/up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2C
Aconvolutional_autoencoder/convolution_decoder/up_sampling2d_2/mul?
Zconvolutional_autoencoder/convolution_decoder/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighborIconvolutional_autoencoder/convolution_decoder/conv2d_6/Relu:activations:0Econvolutional_autoencoder/convolution_decoder/up_sampling2d_2/mul:z:0*
T0*/
_output_shapes
:?????????  @*
half_pixel_centers(2\
Zconvolutional_autoencoder/convolution_decoder/up_sampling2d_2/resize/ResizeNearestNeighbor?
Lconvolutional_autoencoder/convolution_decoder/conv2d_7/Conv2D/ReadVariableOpReadVariableOpUconvolutional_autoencoder_convolution_decoder_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02N
Lconvolutional_autoencoder/convolution_decoder/conv2d_7/Conv2D/ReadVariableOp?
=convolutional_autoencoder/convolution_decoder/conv2d_7/Conv2DConv2Dkconvolutional_autoencoder/convolution_decoder/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0Tconvolutional_autoencoder/convolution_decoder/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2?
=convolutional_autoencoder/convolution_decoder/conv2d_7/Conv2D?
Mconvolutional_autoencoder/convolution_decoder/conv2d_7/BiasAdd/ReadVariableOpReadVariableOpVconvolutional_autoencoder_convolution_decoder_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02O
Mconvolutional_autoencoder/convolution_decoder/conv2d_7/BiasAdd/ReadVariableOp?
>convolutional_autoencoder/convolution_decoder/conv2d_7/BiasAddBiasAddFconvolutional_autoencoder/convolution_decoder/conv2d_7/Conv2D:output:0Uconvolutional_autoencoder/convolution_decoder/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2@
>convolutional_autoencoder/convolution_decoder/conv2d_7/BiasAdd?
;convolutional_autoencoder/convolution_decoder/conv2d_7/ReluReluGconvolutional_autoencoder/convolution_decoder/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2=
;convolutional_autoencoder/convolution_decoder/conv2d_7/Relu?
Cconvolutional_autoencoder/convolution_decoder/up_sampling2d_3/ShapeShapeIconvolutional_autoencoder/convolution_decoder/conv2d_7/Relu:activations:0*
T0*
_output_shapes
:2E
Cconvolutional_autoencoder/convolution_decoder/up_sampling2d_3/Shape?
Qconvolutional_autoencoder/convolution_decoder/up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2S
Qconvolutional_autoencoder/convolution_decoder/up_sampling2d_3/strided_slice/stack?
Sconvolutional_autoencoder/convolution_decoder/up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2U
Sconvolutional_autoencoder/convolution_decoder/up_sampling2d_3/strided_slice/stack_1?
Sconvolutional_autoencoder/convolution_decoder/up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2U
Sconvolutional_autoencoder/convolution_decoder/up_sampling2d_3/strided_slice/stack_2?
Kconvolutional_autoencoder/convolution_decoder/up_sampling2d_3/strided_sliceStridedSliceLconvolutional_autoencoder/convolution_decoder/up_sampling2d_3/Shape:output:0Zconvolutional_autoencoder/convolution_decoder/up_sampling2d_3/strided_slice/stack:output:0\convolutional_autoencoder/convolution_decoder/up_sampling2d_3/strided_slice/stack_1:output:0\convolutional_autoencoder/convolution_decoder/up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2M
Kconvolutional_autoencoder/convolution_decoder/up_sampling2d_3/strided_slice?
Cconvolutional_autoencoder/convolution_decoder/up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2E
Cconvolutional_autoencoder/convolution_decoder/up_sampling2d_3/Const?
Aconvolutional_autoencoder/convolution_decoder/up_sampling2d_3/mulMulTconvolutional_autoencoder/convolution_decoder/up_sampling2d_3/strided_slice:output:0Lconvolutional_autoencoder/convolution_decoder/up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2C
Aconvolutional_autoencoder/convolution_decoder/up_sampling2d_3/mul?
Zconvolutional_autoencoder/convolution_decoder/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighborIconvolutional_autoencoder/convolution_decoder/conv2d_7/Relu:activations:0Econvolutional_autoencoder/convolution_decoder/up_sampling2d_3/mul:z:0*
T0*/
_output_shapes
:?????????@@ *
half_pixel_centers(2\
Zconvolutional_autoencoder/convolution_decoder/up_sampling2d_3/resize/ResizeNearestNeighbor?
Lconvolutional_autoencoder/convolution_decoder/conv2d_8/Conv2D/ReadVariableOpReadVariableOpUconvolutional_autoencoder_convolution_decoder_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02N
Lconvolutional_autoencoder/convolution_decoder/conv2d_8/Conv2D/ReadVariableOp?
=convolutional_autoencoder/convolution_decoder/conv2d_8/Conv2DConv2Dkconvolutional_autoencoder/convolution_decoder/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0Tconvolutional_autoencoder/convolution_decoder/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2?
=convolutional_autoencoder/convolution_decoder/conv2d_8/Conv2D?
Mconvolutional_autoencoder/convolution_decoder/conv2d_8/BiasAdd/ReadVariableOpReadVariableOpVconvolutional_autoencoder_convolution_decoder_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02O
Mconvolutional_autoencoder/convolution_decoder/conv2d_8/BiasAdd/ReadVariableOp?
>convolutional_autoencoder/convolution_decoder/conv2d_8/BiasAddBiasAddFconvolutional_autoencoder/convolution_decoder/conv2d_8/Conv2D:output:0Uconvolutional_autoencoder/convolution_decoder/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2@
>convolutional_autoencoder/convolution_decoder/conv2d_8/BiasAdd?
IdentityIdentityGconvolutional_autoencoder/convolution_decoder/conv2d_8/BiasAdd:output:0N^convolutional_autoencoder/convolution_decoder/conv2d_4/BiasAdd/ReadVariableOpM^convolutional_autoencoder/convolution_decoder/conv2d_4/Conv2D/ReadVariableOpN^convolutional_autoencoder/convolution_decoder/conv2d_5/BiasAdd/ReadVariableOpM^convolutional_autoencoder/convolution_decoder/conv2d_5/Conv2D/ReadVariableOpN^convolutional_autoencoder/convolution_decoder/conv2d_6/BiasAdd/ReadVariableOpM^convolutional_autoencoder/convolution_decoder/conv2d_6/Conv2D/ReadVariableOpN^convolutional_autoencoder/convolution_decoder/conv2d_7/BiasAdd/ReadVariableOpM^convolutional_autoencoder/convolution_decoder/conv2d_7/Conv2D/ReadVariableOpN^convolutional_autoencoder/convolution_decoder/conv2d_8/BiasAdd/ReadVariableOpM^convolutional_autoencoder/convolution_decoder/conv2d_8/Conv2D/ReadVariableOpM^convolutional_autoencoder/convolution_decoder/dense_1/BiasAdd/ReadVariableOpL^convolutional_autoencoder/convolution_decoder/dense_1/MatMul/ReadVariableOpL^convolutional_autoencoder/convolution_encoder/conv2d/BiasAdd/ReadVariableOpK^convolutional_autoencoder/convolution_encoder/conv2d/Conv2D/ReadVariableOpN^convolutional_autoencoder/convolution_encoder/conv2d_1/BiasAdd/ReadVariableOpM^convolutional_autoencoder/convolution_encoder/conv2d_1/Conv2D/ReadVariableOpN^convolutional_autoencoder/convolution_encoder/conv2d_2/BiasAdd/ReadVariableOpM^convolutional_autoencoder/convolution_encoder/conv2d_2/Conv2D/ReadVariableOpN^convolutional_autoencoder/convolution_encoder/conv2d_3/BiasAdd/ReadVariableOpM^convolutional_autoencoder/convolution_encoder/conv2d_3/Conv2D/ReadVariableOpK^convolutional_autoencoder/convolution_encoder/dense/BiasAdd/ReadVariableOpJ^convolutional_autoencoder/convolution_encoder/dense/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????@@::::::::::::::::::::::2?
Mconvolutional_autoencoder/convolution_decoder/conv2d_4/BiasAdd/ReadVariableOpMconvolutional_autoencoder/convolution_decoder/conv2d_4/BiasAdd/ReadVariableOp2?
Lconvolutional_autoencoder/convolution_decoder/conv2d_4/Conv2D/ReadVariableOpLconvolutional_autoencoder/convolution_decoder/conv2d_4/Conv2D/ReadVariableOp2?
Mconvolutional_autoencoder/convolution_decoder/conv2d_5/BiasAdd/ReadVariableOpMconvolutional_autoencoder/convolution_decoder/conv2d_5/BiasAdd/ReadVariableOp2?
Lconvolutional_autoencoder/convolution_decoder/conv2d_5/Conv2D/ReadVariableOpLconvolutional_autoencoder/convolution_decoder/conv2d_5/Conv2D/ReadVariableOp2?
Mconvolutional_autoencoder/convolution_decoder/conv2d_6/BiasAdd/ReadVariableOpMconvolutional_autoencoder/convolution_decoder/conv2d_6/BiasAdd/ReadVariableOp2?
Lconvolutional_autoencoder/convolution_decoder/conv2d_6/Conv2D/ReadVariableOpLconvolutional_autoencoder/convolution_decoder/conv2d_6/Conv2D/ReadVariableOp2?
Mconvolutional_autoencoder/convolution_decoder/conv2d_7/BiasAdd/ReadVariableOpMconvolutional_autoencoder/convolution_decoder/conv2d_7/BiasAdd/ReadVariableOp2?
Lconvolutional_autoencoder/convolution_decoder/conv2d_7/Conv2D/ReadVariableOpLconvolutional_autoencoder/convolution_decoder/conv2d_7/Conv2D/ReadVariableOp2?
Mconvolutional_autoencoder/convolution_decoder/conv2d_8/BiasAdd/ReadVariableOpMconvolutional_autoencoder/convolution_decoder/conv2d_8/BiasAdd/ReadVariableOp2?
Lconvolutional_autoencoder/convolution_decoder/conv2d_8/Conv2D/ReadVariableOpLconvolutional_autoencoder/convolution_decoder/conv2d_8/Conv2D/ReadVariableOp2?
Lconvolutional_autoencoder/convolution_decoder/dense_1/BiasAdd/ReadVariableOpLconvolutional_autoencoder/convolution_decoder/dense_1/BiasAdd/ReadVariableOp2?
Kconvolutional_autoencoder/convolution_decoder/dense_1/MatMul/ReadVariableOpKconvolutional_autoencoder/convolution_decoder/dense_1/MatMul/ReadVariableOp2?
Kconvolutional_autoencoder/convolution_encoder/conv2d/BiasAdd/ReadVariableOpKconvolutional_autoencoder/convolution_encoder/conv2d/BiasAdd/ReadVariableOp2?
Jconvolutional_autoencoder/convolution_encoder/conv2d/Conv2D/ReadVariableOpJconvolutional_autoencoder/convolution_encoder/conv2d/Conv2D/ReadVariableOp2?
Mconvolutional_autoencoder/convolution_encoder/conv2d_1/BiasAdd/ReadVariableOpMconvolutional_autoencoder/convolution_encoder/conv2d_1/BiasAdd/ReadVariableOp2?
Lconvolutional_autoencoder/convolution_encoder/conv2d_1/Conv2D/ReadVariableOpLconvolutional_autoencoder/convolution_encoder/conv2d_1/Conv2D/ReadVariableOp2?
Mconvolutional_autoencoder/convolution_encoder/conv2d_2/BiasAdd/ReadVariableOpMconvolutional_autoencoder/convolution_encoder/conv2d_2/BiasAdd/ReadVariableOp2?
Lconvolutional_autoencoder/convolution_encoder/conv2d_2/Conv2D/ReadVariableOpLconvolutional_autoencoder/convolution_encoder/conv2d_2/Conv2D/ReadVariableOp2?
Mconvolutional_autoencoder/convolution_encoder/conv2d_3/BiasAdd/ReadVariableOpMconvolutional_autoencoder/convolution_encoder/conv2d_3/BiasAdd/ReadVariableOp2?
Lconvolutional_autoencoder/convolution_encoder/conv2d_3/Conv2D/ReadVariableOpLconvolutional_autoencoder/convolution_encoder/conv2d_3/Conv2D/ReadVariableOp2?
Jconvolutional_autoencoder/convolution_encoder/dense/BiasAdd/ReadVariableOpJconvolutional_autoencoder/convolution_encoder/dense/BiasAdd/ReadVariableOp2?
Iconvolutional_autoencoder/convolution_encoder/dense/MatMul/ReadVariableOpIconvolutional_autoencoder/convolution_encoder/dense/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?

?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_10357

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_10661

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
{
&__inference_conv2d_layer_call_fn_11058

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_103302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
}
(__inference_conv2d_5_layer_call_fn_11158

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_106612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_10564

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_10957

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
C
'__inference_reshape_layer_call_fn_11019

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_106142
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_6_layer_call_and_return_conditional_losses_10689

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_10545

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_11089

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????@@D
output_18
StatefulPartitionedCall:0?????????@@tensorflow/serving/predict:??
?
	optimizer
encoder
decoder
regularization_losses
trainable_variables
	variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "ConvolutionalAutoencoder", "name": "convolutional_autoencoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"image_size": 64, "code_dim": 8, "depth": 4}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 64, 64, 1]}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ConvolutionalAutoencoder", "config": {"image_size": 64, "code_dim": 8, "depth": 4}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
	iter


beta_1

beta_2
	decay
learning_ratem?m? m?!m?"m?#m?$m?%m?&m?'m?(m?)m?*m?+m?,m?-m?.m?/m?0m?1m?2m?3m?v?v? v?!v?"v?#v?$v?%v?&v?'v?(v?)v?*v?+v?,v?-v?.v?/v?0v?1v?2v?3v?"
	optimizer
?
	convs
flat
dense_1
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "ConvolutionEncoder", "name": "convolution_encoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 64, 64, 1]}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ConvolutionEncoder"}}
?
dense_1
reshape
	convs
	upsamples

final_conv
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "ConvolutionDecoder", "name": "convolution_decoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 8]}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ConvolutionDecoder"}}
 "
trackable_list_wrapper
?
0
1
 2
!3
"4
#5
$6
%7
&8
'9
(10
)11
*12
+13
,14
-15
.16
/17
018
119
220
321"
trackable_list_wrapper
?
0
1
 2
!3
"4
#5
$6
%7
&8
'9
(10
)11
*12
+13
,14
-15
.16
/17
018
119
220
321"
trackable_list_wrapper
?

4layers
regularization_losses
trainable_variables
5layer_metrics
6non_trainable_variables
7metrics
8layer_regularization_losses
	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
<
90
:1
;2
<3"
trackable_list_wrapper
?
=regularization_losses
>trainable_variables
?	variables
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

&kernel
'bias
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1152}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 1152]}}
 "
trackable_list_wrapper
f
0
1
 2
!3
"4
#5
$6
%7
&8
'9"
trackable_list_wrapper
f
0
1
 2
!3
"4
#5
$6
%7
&8
'9"
trackable_list_wrapper
?

Elayers
regularization_losses
trainable_variables
Flayer_metrics
Gnon_trainable_variables
Hmetrics
Ilayer_regularization_losses
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

(kernel
)bias
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 8]}}
?
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4, 16]}}}
<
R0
S1
T2
U3"
trackable_list_wrapper
<
V0
W1
X2
Y3"
trackable_list_wrapper
?	

2kernel
3bias
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 64, 64, 32]}}
 "
trackable_list_wrapper
v
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311"
trackable_list_wrapper
v
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311"
trackable_list_wrapper
?

^layers
regularization_losses
trainable_variables
_layer_metrics
`non_trainable_variables
ametrics
blayer_regularization_losses
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
U:S2;convolutional_autoencoder/convolution_encoder/conv2d/kernel
G:E29convolutional_autoencoder/convolution_encoder/conv2d/bias
W:U 2=convolutional_autoencoder/convolution_encoder/conv2d_1/kernel
I:G 2;convolutional_autoencoder/convolution_encoder/conv2d_1/bias
W:U @2=convolutional_autoencoder/convolution_encoder/conv2d_2/kernel
I:G@2;convolutional_autoencoder/convolution_encoder/conv2d_2/bias
X:V@?2=convolutional_autoencoder/convolution_encoder/conv2d_3/kernel
J:H?2;convolutional_autoencoder/convolution_encoder/conv2d_3/bias
M:K	?	2:convolutional_autoencoder/convolution_encoder/dense/kernel
F:D28convolutional_autoencoder/convolution_encoder/dense/bias
O:M	?2<convolutional_autoencoder/convolution_decoder/dense_1/kernel
I:G?2:convolutional_autoencoder/convolution_decoder/dense_1/bias
X:V?2=convolutional_autoencoder/convolution_decoder/conv2d_4/kernel
J:H?2;convolutional_autoencoder/convolution_decoder/conv2d_4/bias
Y:W??2=convolutional_autoencoder/convolution_decoder/conv2d_5/kernel
J:H?2;convolutional_autoencoder/convolution_decoder/conv2d_5/bias
X:V?@2=convolutional_autoencoder/convolution_decoder/conv2d_6/kernel
I:G@2;convolutional_autoencoder/convolution_decoder/conv2d_6/bias
W:U@ 2=convolutional_autoencoder/convolution_decoder/conv2d_7/kernel
I:G 2;convolutional_autoencoder/convolution_decoder/conv2d_7/bias
W:U 2=convolutional_autoencoder/convolution_decoder/conv2d_8/kernel
I:G2;convolutional_autoencoder/convolution_decoder/conv2d_8/bias
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
c0"
trackable_list_wrapper
 "
trackable_list_wrapper
?	

kernel
bias
dregularization_losses
etrainable_variables
f	variables
g	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 64, 64, 1]}}
?	

 kernel
!bias
hregularization_losses
itrainable_variables
j	variables
k	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 31, 31, 16]}}
?	

"kernel
#bias
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 15, 15, 32]}}
?	

$kernel
%bias
pregularization_losses
qtrainable_variables
r	variables
s	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 7, 7, 64]}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

tlayers
=regularization_losses
>trainable_variables
ulayer_metrics
vnon_trainable_variables
wmetrics
xlayer_regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
?

ylayers
Aregularization_losses
Btrainable_variables
zlayer_metrics
{non_trainable_variables
|metrics
}layer_regularization_losses
C	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
J
90
:1
;2
<3
4
5"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?

~layers
Jregularization_losses
Ktrainable_variables
layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
L	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
Nregularization_losses
Otrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
P	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?	

*kernel
+bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 4, 4, 16]}}
?	

,kernel
-bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 8, 8, 256]}}
?	

.kernel
/bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 16, 16, 128]}}
?	

0kernel
1bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 32, 32, 64]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
?
?layers
Zregularization_losses
[trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
\	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
n
0
1
R2
S3
T4
U5
V6
W7
X8
Y9
10"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
?layers
dregularization_losses
etrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
f	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
?
?layers
hregularization_losses
itrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
j	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?
?layers
lregularization_losses
mtrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
n	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
?
?layers
pregularization_losses
qtrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
r	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
?
?layers
?regularization_losses
?trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
?
?layers
?regularization_losses
?trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
?layers
?regularization_losses
?trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
?
?layers
?regularization_losses
?trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?regularization_losses
?trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?regularization_losses
?trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?regularization_losses
?trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?regularization_losses
?trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Z:X2BAdam/convolutional_autoencoder/convolution_encoder/conv2d/kernel/m
L:J2@Adam/convolutional_autoencoder/convolution_encoder/conv2d/bias/m
\:Z 2DAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/kernel/m
N:L 2BAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/bias/m
\:Z @2DAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/kernel/m
N:L@2BAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/bias/m
]:[@?2DAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/kernel/m
O:M?2BAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/bias/m
R:P	?	2AAdam/convolutional_autoencoder/convolution_encoder/dense/kernel/m
K:I2?Adam/convolutional_autoencoder/convolution_encoder/dense/bias/m
T:R	?2CAdam/convolutional_autoencoder/convolution_decoder/dense_1/kernel/m
N:L?2AAdam/convolutional_autoencoder/convolution_decoder/dense_1/bias/m
]:[?2DAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/kernel/m
O:M?2BAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/bias/m
^:\??2DAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/kernel/m
O:M?2BAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/bias/m
]:[?@2DAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/kernel/m
N:L@2BAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/bias/m
\:Z@ 2DAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/kernel/m
N:L 2BAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/bias/m
\:Z 2DAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/kernel/m
N:L2BAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/bias/m
Z:X2BAdam/convolutional_autoencoder/convolution_encoder/conv2d/kernel/v
L:J2@Adam/convolutional_autoencoder/convolution_encoder/conv2d/bias/v
\:Z 2DAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/kernel/v
N:L 2BAdam/convolutional_autoencoder/convolution_encoder/conv2d_1/bias/v
\:Z @2DAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/kernel/v
N:L@2BAdam/convolutional_autoencoder/convolution_encoder/conv2d_2/bias/v
]:[@?2DAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/kernel/v
O:M?2BAdam/convolutional_autoencoder/convolution_encoder/conv2d_3/bias/v
R:P	?	2AAdam/convolutional_autoencoder/convolution_encoder/dense/kernel/v
K:I2?Adam/convolutional_autoencoder/convolution_encoder/dense/bias/v
T:R	?2CAdam/convolutional_autoencoder/convolution_decoder/dense_1/kernel/v
N:L?2AAdam/convolutional_autoencoder/convolution_decoder/dense_1/bias/v
]:[?2DAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/kernel/v
O:M?2BAdam/convolutional_autoencoder/convolution_decoder/conv2d_4/bias/v
^:\??2DAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/kernel/v
O:M?2BAdam/convolutional_autoencoder/convolution_decoder/conv2d_5/bias/v
]:[?@2DAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/kernel/v
N:L@2BAdam/convolutional_autoencoder/convolution_decoder/conv2d_6/bias/v
\:Z@ 2DAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/kernel/v
N:L 2BAdam/convolutional_autoencoder/convolution_decoder/conv2d_7/bias/v
\:Z 2DAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/kernel/v
N:L2BAdam/convolutional_autoencoder/convolution_decoder/conv2d_8/bias/v
?2?
9__inference_convolutional_autoencoder_layer_call_fn_10892?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????@@
?2?
 __inference__wrapped_model_10315?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????@@
?2?
T__inference_convolutional_autoencoder_layer_call_and_return_conditional_losses_10842?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????@@
?2?
3__inference_convolution_encoder_layer_call_fn_10494?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????@@
?2?
N__inference_convolution_encoder_layer_call_and_return_conditional_losses_10468?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????@@
?2?
3__inference_convolution_decoder_layer_call_fn_10791?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
N__inference_convolution_decoder_layer_call_and_return_conditional_losses_10761?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?B?
#__inference_signature_wrapper_10951input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_flatten_layer_call_fn_10962?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_flatten_layer_call_and_return_conditional_losses_10957?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_10981?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_10972?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_11000?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_10991?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_reshape_layer_call_fn_11019?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_reshape_layer_call_and_return_conditional_losses_11014?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_8_layer_call_fn_11038?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_8_layer_call_and_return_conditional_losses_11029?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_conv2d_layer_call_fn_11058?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv2d_layer_call_and_return_conditional_losses_11049?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_1_layer_call_fn_11078?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_11069?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_2_layer_call_fn_11098?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_11089?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_3_layer_call_fn_11118?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_11109?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_4_layer_call_fn_11138?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_11129?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_5_layer_call_fn_11158?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_11149?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_6_layer_call_fn_11178?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_6_layer_call_and_return_conditional_losses_11169?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_7_layer_call_fn_11198?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_7_layer_call_and_return_conditional_losses_11189?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_up_sampling2d_layer_call_fn_10513?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_10507?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
/__inference_up_sampling2d_1_layer_call_fn_10532?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_10526?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
/__inference_up_sampling2d_2_layer_call_fn_10551?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
J__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_10545?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
/__inference_up_sampling2d_3_layer_call_fn_10570?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_10564?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84?????????????????????????????????????
 __inference__wrapped_model_10315? !"#$%&'()*+,-./01238?5
.?+
)?&
input_1?????????@@
? ";?8
6
output_1*?'
output_1?????????@@?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_11069l !7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
(__inference_conv2d_1_layer_call_fn_11078_ !7?4
-?*
(?%
inputs?????????
? " ?????????? ?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_11089l"#7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
(__inference_conv2d_2_layer_call_fn_11098_"#7?4
-?*
(?%
inputs????????? 
? " ??????????@?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_11109m$%7?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0??????????
? ?
(__inference_conv2d_3_layer_call_fn_11118`$%7?4
-?*
(?%
inputs?????????@
? "!????????????
C__inference_conv2d_4_layer_call_and_return_conditional_losses_11129m*+7?4
-?*
(?%
inputs?????????
? ".?+
$?!
0??????????
? ?
(__inference_conv2d_4_layer_call_fn_11138`*+7?4
-?*
(?%
inputs?????????
? "!????????????
C__inference_conv2d_5_layer_call_and_return_conditional_losses_11149?,-J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
(__inference_conv2d_5_layer_call_fn_11158?,-J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
C__inference_conv2d_6_layer_call_and_return_conditional_losses_11169?./J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
(__inference_conv2d_6_layer_call_fn_11178?./J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+???????????????????????????@?
C__inference_conv2d_7_layer_call_and_return_conditional_losses_11189?01I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
(__inference_conv2d_7_layer_call_fn_11198?01I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
C__inference_conv2d_8_layer_call_and_return_conditional_losses_11029?23I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
(__inference_conv2d_8_layer_call_fn_11038?23I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
A__inference_conv2d_layer_call_and_return_conditional_losses_11049l7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????
? ?
&__inference_conv2d_layer_call_fn_11058_7?4
-?*
(?%
inputs?????????@@
? " ???????????
N__inference_convolution_decoder_layer_call_and_return_conditional_losses_10761?()*+,-./01230?-
&?#
!?
input_1?????????
? "??<
5?2
0+???????????????????????????
? ?
3__inference_convolution_decoder_layer_call_fn_10791t()*+,-./01230?-
&?#
!?
input_1?????????
? "2?/+????????????????????????????
N__inference_convolution_encoder_layer_call_and_return_conditional_losses_10468m
 !"#$%&'8?5
.?+
)?&
input_1?????????@@
? "%?"
?
0?????????
? ?
3__inference_convolution_encoder_layer_call_fn_10494`
 !"#$%&'8?5
.?+
)?&
input_1?????????@@
? "???????????
T__inference_convolutional_autoencoder_layer_call_and_return_conditional_losses_10842? !"#$%&'()*+,-./01238?5
.?+
)?&
input_1?????????@@
? "??<
5?2
0+???????????????????????????
? ?
9__inference_convolutional_autoencoder_layer_call_fn_10892? !"#$%&'()*+,-./01238?5
.?+
)?&
input_1?????????@@
? "2?/+????????????????????????????
B__inference_dense_1_layer_call_and_return_conditional_losses_10991]()/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? {
'__inference_dense_1_layer_call_fn_11000P()/?,
%?"
 ?
inputs?????????
? "????????????
@__inference_dense_layer_call_and_return_conditional_losses_10972]&'0?-
&?#
!?
inputs??????????	
? "%?"
?
0?????????
? y
%__inference_dense_layer_call_fn_10981P&'0?-
&?#
!?
inputs??????????	
? "???????????
B__inference_flatten_layer_call_and_return_conditional_losses_10957b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????	
? ?
'__inference_flatten_layer_call_fn_10962U8?5
.?+
)?&
inputs??????????
? "???????????	?
B__inference_reshape_layer_call_and_return_conditional_losses_11014a0?-
&?#
!?
inputs??????????
? "-?*
#? 
0?????????
? 
'__inference_reshape_layer_call_fn_11019T0?-
&?#
!?
inputs??????????
? " ???????????
#__inference_signature_wrapper_10951? !"#$%&'()*+,-./0123C?@
? 
9?6
4
input_1)?&
input_1?????????@@";?8
6
output_1*?'
output_1?????????@@?
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_10526?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_up_sampling2d_1_layer_call_fn_10532?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_10545?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_up_sampling2d_2_layer_call_fn_10551?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_10564?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_up_sampling2d_3_layer_call_fn_10570?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_10507?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
-__inference_up_sampling2d_layer_call_fn_10513?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????