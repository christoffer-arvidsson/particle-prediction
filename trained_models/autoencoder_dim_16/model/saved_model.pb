??
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
Aconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*R
shared_nameCAconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/kernel
?
Uconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/kernel/Read/ReadVariableOpReadVariableOpAconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/kernel*&
_output_shapes
:*
dtype0
?
?convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/bias
?
Sconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/bias/Read/ReadVariableOpReadVariableOp?convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/bias*
_output_shapes
:*
dtype0
?
Bconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *S
shared_nameDBconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/kernel
?
Vconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/kernel*&
_output_shapes
: *
dtype0
?
@convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/bias
?
Tconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/bias*
_output_shapes
: *
dtype0
?
Bconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*S
shared_nameDBconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/kernel
?
Vconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/kernel*&
_output_shapes
: @*
dtype0
?
@convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*Q
shared_nameB@convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/bias
?
Tconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/bias*
_output_shapes
:@*
dtype0
?
Bconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*S
shared_nameDBconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/kernel
?
Vconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/kernel*'
_output_shapes
:@?*
dtype0
?
@convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*Q
shared_nameB@convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/bias
?
Tconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/bias*
_output_shapes	
:?*
dtype0
?
@convolutional_autoencoder_1/convolution_encoder_1/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	*Q
shared_nameB@convolutional_autoencoder_1/convolution_encoder_1/dense_2/kernel
?
Tconvolutional_autoencoder_1/convolution_encoder_1/dense_2/kernel/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_1/convolution_encoder_1/dense_2/kernel*
_output_shapes
:	?	*
dtype0
?
>convolutional_autoencoder_1/convolution_encoder_1/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*O
shared_name@>convolutional_autoencoder_1/convolution_encoder_1/dense_2/bias
?
Rconvolutional_autoencoder_1/convolution_encoder_1/dense_2/bias/Read/ReadVariableOpReadVariableOp>convolutional_autoencoder_1/convolution_encoder_1/dense_2/bias*
_output_shapes
:*
dtype0
?
@convolutional_autoencoder_1/convolution_decoder_1/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*Q
shared_nameB@convolutional_autoencoder_1/convolution_decoder_1/dense_3/kernel
?
Tconvolutional_autoencoder_1/convolution_decoder_1/dense_3/kernel/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_1/convolution_decoder_1/dense_3/kernel*
_output_shapes
:	?*
dtype0
?
>convolutional_autoencoder_1/convolution_decoder_1/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*O
shared_name@>convolutional_autoencoder_1/convolution_decoder_1/dense_3/bias
?
Rconvolutional_autoencoder_1/convolution_decoder_1/dense_3/bias/Read/ReadVariableOpReadVariableOp>convolutional_autoencoder_1/convolution_decoder_1/dense_3/bias*
_output_shapes	
:?*
dtype0
?
Bconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*S
shared_nameDBconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/kernel
?
Vconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/kernel*'
_output_shapes
:?*
dtype0
?
@convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*Q
shared_nameB@convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/bias
?
Tconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/bias*
_output_shapes	
:?*
dtype0
?
Bconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*S
shared_nameDBconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/kernel
?
Vconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/kernel*(
_output_shapes
:??*
dtype0
?
@convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*Q
shared_nameB@convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/bias
?
Tconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/bias*
_output_shapes	
:?*
dtype0
?
Bconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*S
shared_nameDBconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/kernel
?
Vconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/kernel*'
_output_shapes
:?@*
dtype0
?
@convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*Q
shared_nameB@convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/bias
?
Tconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/bias*
_output_shapes
:@*
dtype0
?
Bconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *S
shared_nameDBconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/kernel
?
Vconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/kernel*&
_output_shapes
:@ *
dtype0
?
@convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/bias
?
Tconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/bias*
_output_shapes
: *
dtype0
?
Bconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *S
shared_nameDBconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/kernel
?
Vconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/kernel*&
_output_shapes
: *
dtype0
?
@convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/bias
?
Tconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/bias*
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
HAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Y
shared_nameJHAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/kernel/m
?
\Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/kernel/m/Read/ReadVariableOpReadVariableOpHAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/kernel/m*&
_output_shapes
:*
dtype0
?
FAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*W
shared_nameHFAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/bias/m
?
ZAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/bias/m/Read/ReadVariableOpReadVariableOpFAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/bias/m*
_output_shapes
:*
dtype0
?
IAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Z
shared_nameKIAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/kernel/m
?
]Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/kernel/m*&
_output_shapes
: *
dtype0
?
GAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/bias/m
?
[Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/bias/m*
_output_shapes
: *
dtype0
?
IAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*Z
shared_nameKIAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/kernel/m
?
]Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/kernel/m*&
_output_shapes
: @*
dtype0
?
GAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*X
shared_nameIGAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/bias/m
?
[Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/bias/m*
_output_shapes
:@*
dtype0
?
IAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*Z
shared_nameKIAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/kernel/m
?
]Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/kernel/m*'
_output_shapes
:@?*
dtype0
?
GAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*X
shared_nameIGAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/bias/m
?
[Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/bias/m*
_output_shapes	
:?*
dtype0
?
GAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	*X
shared_nameIGAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/kernel/m
?
[Adam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/kernel/m*
_output_shapes
:	?	*
dtype0
?
EAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*V
shared_nameGEAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/bias/m
?
YAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/bias/m/Read/ReadVariableOpReadVariableOpEAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/bias/m*
_output_shapes
:*
dtype0
?
GAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*X
shared_nameIGAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/kernel/m
?
[Adam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/kernel/m*
_output_shapes
:	?*
dtype0
?
EAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*V
shared_nameGEAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/bias/m
?
YAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/bias/m/Read/ReadVariableOpReadVariableOpEAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/bias/m*
_output_shapes	
:?*
dtype0
?
IAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*Z
shared_nameKIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/kernel/m
?
]Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/kernel/m*'
_output_shapes
:?*
dtype0
?
GAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*X
shared_nameIGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/bias/m
?
[Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/bias/m*
_output_shapes	
:?*
dtype0
?
IAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*Z
shared_nameKIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/kernel/m
?
]Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/kernel/m*(
_output_shapes
:??*
dtype0
?
GAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*X
shared_nameIGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/bias/m
?
[Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/bias/m*
_output_shapes	
:?*
dtype0
?
IAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*Z
shared_nameKIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/kernel/m
?
]Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/kernel/m*'
_output_shapes
:?@*
dtype0
?
GAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*X
shared_nameIGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/bias/m
?
[Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/bias/m*
_output_shapes
:@*
dtype0
?
IAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *Z
shared_nameKIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/kernel/m
?
]Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/kernel/m*&
_output_shapes
:@ *
dtype0
?
GAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/bias/m
?
[Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/bias/m*
_output_shapes
: *
dtype0
?
IAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Z
shared_nameKIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/kernel/m
?
]Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/kernel/m*&
_output_shapes
: *
dtype0
?
GAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*X
shared_nameIGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/bias/m
?
[Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/bias/m*
_output_shapes
:*
dtype0
?
HAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Y
shared_nameJHAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/kernel/v
?
\Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/kernel/v/Read/ReadVariableOpReadVariableOpHAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/kernel/v*&
_output_shapes
:*
dtype0
?
FAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*W
shared_nameHFAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/bias/v
?
ZAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/bias/v/Read/ReadVariableOpReadVariableOpFAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/bias/v*
_output_shapes
:*
dtype0
?
IAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Z
shared_nameKIAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/kernel/v
?
]Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/kernel/v*&
_output_shapes
: *
dtype0
?
GAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/bias/v
?
[Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/bias/v*
_output_shapes
: *
dtype0
?
IAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*Z
shared_nameKIAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/kernel/v
?
]Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/kernel/v*&
_output_shapes
: @*
dtype0
?
GAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*X
shared_nameIGAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/bias/v
?
[Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/bias/v*
_output_shapes
:@*
dtype0
?
IAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*Z
shared_nameKIAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/kernel/v
?
]Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/kernel/v*'
_output_shapes
:@?*
dtype0
?
GAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*X
shared_nameIGAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/bias/v
?
[Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/bias/v*
_output_shapes	
:?*
dtype0
?
GAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	*X
shared_nameIGAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/kernel/v
?
[Adam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/kernel/v*
_output_shapes
:	?	*
dtype0
?
EAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*V
shared_nameGEAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/bias/v
?
YAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/bias/v/Read/ReadVariableOpReadVariableOpEAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/bias/v*
_output_shapes
:*
dtype0
?
GAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*X
shared_nameIGAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/kernel/v
?
[Adam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/kernel/v*
_output_shapes
:	?*
dtype0
?
EAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*V
shared_nameGEAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/bias/v
?
YAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/bias/v/Read/ReadVariableOpReadVariableOpEAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/bias/v*
_output_shapes	
:?*
dtype0
?
IAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*Z
shared_nameKIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/kernel/v
?
]Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/kernel/v*'
_output_shapes
:?*
dtype0
?
GAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*X
shared_nameIGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/bias/v
?
[Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/bias/v*
_output_shapes	
:?*
dtype0
?
IAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*Z
shared_nameKIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/kernel/v
?
]Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/kernel/v*(
_output_shapes
:??*
dtype0
?
GAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*X
shared_nameIGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/bias/v
?
[Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/bias/v*
_output_shapes	
:?*
dtype0
?
IAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*Z
shared_nameKIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/kernel/v
?
]Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/kernel/v*'
_output_shapes
:?@*
dtype0
?
GAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*X
shared_nameIGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/bias/v
?
[Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/bias/v*
_output_shapes
:@*
dtype0
?
IAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *Z
shared_nameKIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/kernel/v
?
]Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/kernel/v*&
_output_shapes
:@ *
dtype0
?
GAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/bias/v
?
[Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/bias/v*
_output_shapes
: *
dtype0
?
IAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Z
shared_nameKIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/kernel/v
?
]Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/kernel/v*&
_output_shapes
: *
dtype0
?
GAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*X
shared_nameIGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/bias/v
?
[Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ǖ
value??B?? B??
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
??
VARIABLE_VALUEAconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_1/convolution_encoder_1/dense_2/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>convolutional_autoencoder_1/convolution_encoder_1/dense_2/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_1/convolution_decoder_1/dense_3/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>convolutional_autoencoder_1/convolution_decoder_1/dense_3/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/kernel1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/bias1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEHAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEEAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEEAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/kernel/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/bias/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/kernel/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/bias/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/kernel/mMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/bias/mMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/kernel/mMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/bias/mMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEHAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEEAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEEAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/kernel/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/bias/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/kernel/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/bias/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/kernel/vMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/bias/vMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/kernel/vMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/bias/vMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????@@*
dtype0*$
shape:?????????@@
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Aconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/kernel?convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/biasBconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/kernel@convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/biasBconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/kernel@convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/biasBconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/kernel@convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/bias@convolutional_autoencoder_1/convolution_encoder_1/dense_2/kernel>convolutional_autoencoder_1/convolution_encoder_1/dense_2/bias@convolutional_autoencoder_1/convolution_decoder_1/dense_3/kernel>convolutional_autoencoder_1/convolution_decoder_1/dense_3/biasBconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/kernel@convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/biasBconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/kernel@convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/biasBconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/kernel@convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/biasBconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/kernel@convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/biasBconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/kernel@convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/bias*"
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
#__inference_signature_wrapper_23408
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?3
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpUconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/kernel/Read/ReadVariableOpSconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/bias/Read/ReadVariableOpVconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/kernel/Read/ReadVariableOpTconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/bias/Read/ReadVariableOpVconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/kernel/Read/ReadVariableOpTconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/bias/Read/ReadVariableOpVconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/kernel/Read/ReadVariableOpTconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/bias/Read/ReadVariableOpTconvolutional_autoencoder_1/convolution_encoder_1/dense_2/kernel/Read/ReadVariableOpRconvolutional_autoencoder_1/convolution_encoder_1/dense_2/bias/Read/ReadVariableOpTconvolutional_autoencoder_1/convolution_decoder_1/dense_3/kernel/Read/ReadVariableOpRconvolutional_autoencoder_1/convolution_decoder_1/dense_3/bias/Read/ReadVariableOpVconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/kernel/Read/ReadVariableOpTconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/bias/Read/ReadVariableOpVconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/kernel/Read/ReadVariableOpTconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/bias/Read/ReadVariableOpVconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/kernel/Read/ReadVariableOpTconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/bias/Read/ReadVariableOpVconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/kernel/Read/ReadVariableOpTconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/bias/Read/ReadVariableOpVconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/kernel/Read/ReadVariableOpTconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp\Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/kernel/m/Read/ReadVariableOpZAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/bias/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/kernel/m/Read/ReadVariableOpYAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/bias/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/kernel/m/Read/ReadVariableOpYAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/bias/m/Read/ReadVariableOp\Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/kernel/v/Read/ReadVariableOpZAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/bias/v/Read/ReadVariableOp]Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/bias/v/Read/ReadVariableOp]Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/bias/v/Read/ReadVariableOp]Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/bias/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/kernel/v/Read/ReadVariableOpYAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/bias/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/kernel/v/Read/ReadVariableOpYAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/bias/v/Read/ReadVariableOp]Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/bias/v/Read/ReadVariableOp]Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/bias/v/Read/ReadVariableOp]Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/bias/v/Read/ReadVariableOp]Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/bias/v/Read/ReadVariableOp]Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/bias/v/Read/ReadVariableOpConst*V
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
__inference__traced_save_23897
?(
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateAconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/kernel?convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/biasBconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/kernel@convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/biasBconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/kernel@convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/biasBconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/kernel@convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/bias@convolutional_autoencoder_1/convolution_encoder_1/dense_2/kernel>convolutional_autoencoder_1/convolution_encoder_1/dense_2/bias@convolutional_autoencoder_1/convolution_decoder_1/dense_3/kernel>convolutional_autoencoder_1/convolution_decoder_1/dense_3/biasBconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/kernel@convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/biasBconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/kernel@convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/biasBconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/kernel@convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/biasBconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/kernel@convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/biasBconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/kernel@convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/biastotalcountHAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/kernel/mFAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/bias/mIAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/kernel/mGAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/bias/mIAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/kernel/mGAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/bias/mIAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/kernel/mGAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/bias/mGAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/kernel/mEAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/bias/mGAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/kernel/mEAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/bias/mIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/kernel/mGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/bias/mIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/kernel/mGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/bias/mIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/kernel/mGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/bias/mIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/kernel/mGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/bias/mIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/kernel/mGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/bias/mHAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/kernel/vFAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/bias/vIAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/kernel/vGAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/bias/vIAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/kernel/vGAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/bias/vIAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/kernel/vGAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/bias/vGAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/kernel/vEAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/bias/vGAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/kernel/vEAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/bias/vIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/kernel/vGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/bias/vIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/kernel/vGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/bias/vIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/kernel/vGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/bias/vIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/kernel/vGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/bias/vIAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/kernel/vGAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/bias/v*U
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
!__inference__traced_restore_24126??
?
f
J__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_23021

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
D__inference_conv2d_13_layer_call_and_return_conditional_losses_23586

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
?
?
V__inference_convolutional_autoencoder_1_layer_call_and_return_conditional_losses_23299
input_1
convolution_encoder_1_23252
convolution_encoder_1_23254
convolution_encoder_1_23256
convolution_encoder_1_23258
convolution_encoder_1_23260
convolution_encoder_1_23262
convolution_encoder_1_23264
convolution_encoder_1_23266
convolution_encoder_1_23268
convolution_encoder_1_23270
convolution_decoder_1_23273
convolution_decoder_1_23275
convolution_decoder_1_23277
convolution_decoder_1_23279
convolution_decoder_1_23281
convolution_decoder_1_23283
convolution_decoder_1_23285
convolution_decoder_1_23287
convolution_decoder_1_23289
convolution_decoder_1_23291
convolution_decoder_1_23293
convolution_decoder_1_23295
identity??-convolution_decoder_1/StatefulPartitionedCall?-convolution_encoder_1/StatefulPartitionedCall?
-convolution_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinput_1convolution_encoder_1_23252convolution_encoder_1_23254convolution_encoder_1_23256convolution_encoder_1_23258convolution_encoder_1_23260convolution_encoder_1_23262convolution_encoder_1_23264convolution_encoder_1_23266convolution_encoder_1_23268convolution_encoder_1_23270*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_convolution_encoder_1_layer_call_and_return_conditional_losses_229252/
-convolution_encoder_1/StatefulPartitionedCall?
-convolution_decoder_1/StatefulPartitionedCallStatefulPartitionedCall6convolution_encoder_1/StatefulPartitionedCall:output:0convolution_decoder_1_23273convolution_decoder_1_23275convolution_decoder_1_23277convolution_decoder_1_23279convolution_decoder_1_23281convolution_decoder_1_23283convolution_decoder_1_23285convolution_decoder_1_23287convolution_decoder_1_23289convolution_decoder_1_23291convolution_decoder_1_23293convolution_decoder_1_23295*
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
GPU 2J 8? *Y
fTRR
P__inference_convolution_decoder_1_layer_call_and_return_conditional_losses_232182/
-convolution_decoder_1/StatefulPartitionedCall?
IdentityIdentity6convolution_decoder_1/StatefulPartitionedCall:output:0.^convolution_decoder_1/StatefulPartitionedCall.^convolution_encoder_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????@@::::::::::::::::::::::2^
-convolution_decoder_1/StatefulPartitionedCall-convolution_decoder_1/StatefulPartitionedCall2^
-convolution_encoder_1/StatefulPartitionedCall-convolution_encoder_1/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?

?
D__inference_conv2d_10_layer_call_and_return_conditional_losses_22814

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
?
5__inference_convolution_encoder_1_layer_call_fn_22951
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
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_convolution_encoder_1_layer_call_and_return_conditional_losses_229252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

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
D__inference_conv2d_14_layer_call_and_return_conditional_losses_23606

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
?
~
)__inference_conv2d_16_layer_call_fn_23655

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
GPU 2J 8? *M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_231742
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
??
?7
__inference__traced_save_23897
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop`
\savev2_convolutional_autoencoder_1_convolution_encoder_1_conv2d_9_kernel_read_readvariableop^
Zsavev2_convolutional_autoencoder_1_convolution_encoder_1_conv2d_9_bias_read_readvariableopa
]savev2_convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_bias_read_readvariableopa
]savev2_convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_bias_read_readvariableopa
]savev2_convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_bias_read_readvariableop_
[savev2_convolutional_autoencoder_1_convolution_encoder_1_dense_2_kernel_read_readvariableop]
Ysavev2_convolutional_autoencoder_1_convolution_encoder_1_dense_2_bias_read_readvariableop_
[savev2_convolutional_autoencoder_1_convolution_decoder_1_dense_3_kernel_read_readvariableop]
Ysavev2_convolutional_autoencoder_1_convolution_decoder_1_dense_3_bias_read_readvariableopa
]savev2_convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_bias_read_readvariableopa
]savev2_convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_bias_read_readvariableopa
]savev2_convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_bias_read_readvariableopa
]savev2_convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_bias_read_readvariableopa
]savev2_convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopg
csavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_9_kernel_m_read_readvariableope
asavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_9_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_bias_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_dense_2_kernel_m_read_readvariableopd
`savev2_adam_convolutional_autoencoder_1_convolution_encoder_1_dense_2_bias_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_dense_3_kernel_m_read_readvariableopd
`savev2_adam_convolutional_autoencoder_1_convolution_decoder_1_dense_3_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_bias_m_read_readvariableopg
csavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_9_kernel_v_read_readvariableope
asavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_9_bias_v_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_bias_v_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_bias_v_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_bias_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_dense_2_kernel_v_read_readvariableopd
`savev2_adam_convolutional_autoencoder_1_convolution_encoder_1_dense_2_bias_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_dense_3_kernel_v_read_readvariableopd
`savev2_adam_convolutional_autoencoder_1_convolution_decoder_1_dense_3_bias_v_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_bias_v_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_bias_v_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_bias_v_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_bias_v_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_bias_v_read_readvariableop
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
SaveV2/shape_and_slices?6
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop\savev2_convolutional_autoencoder_1_convolution_encoder_1_conv2d_9_kernel_read_readvariableopZsavev2_convolutional_autoencoder_1_convolution_encoder_1_conv2d_9_bias_read_readvariableop]savev2_convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_kernel_read_readvariableop[savev2_convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_bias_read_readvariableop]savev2_convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_kernel_read_readvariableop[savev2_convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_bias_read_readvariableop]savev2_convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_kernel_read_readvariableop[savev2_convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_bias_read_readvariableop[savev2_convolutional_autoencoder_1_convolution_encoder_1_dense_2_kernel_read_readvariableopYsavev2_convolutional_autoencoder_1_convolution_encoder_1_dense_2_bias_read_readvariableop[savev2_convolutional_autoencoder_1_convolution_decoder_1_dense_3_kernel_read_readvariableopYsavev2_convolutional_autoencoder_1_convolution_decoder_1_dense_3_bias_read_readvariableop]savev2_convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_kernel_read_readvariableop[savev2_convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_bias_read_readvariableop]savev2_convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_kernel_read_readvariableop[savev2_convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_bias_read_readvariableop]savev2_convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_kernel_read_readvariableop[savev2_convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_bias_read_readvariableop]savev2_convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_kernel_read_readvariableop[savev2_convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_bias_read_readvariableop]savev2_convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_kernel_read_readvariableop[savev2_convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopcsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_9_kernel_m_read_readvariableopasavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_9_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_bias_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_dense_2_kernel_m_read_readvariableop`savev2_adam_convolutional_autoencoder_1_convolution_encoder_1_dense_2_bias_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_dense_3_kernel_m_read_readvariableop`savev2_adam_convolutional_autoencoder_1_convolution_decoder_1_dense_3_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_bias_m_read_readvariableopcsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_9_kernel_v_read_readvariableopasavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_9_bias_v_read_readvariableopdsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_bias_v_read_readvariableopdsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_bias_v_read_readvariableopdsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_bias_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_1_convolution_encoder_1_dense_2_kernel_v_read_readvariableop`savev2_adam_convolutional_autoencoder_1_convolution_encoder_1_dense_2_bias_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_dense_3_kernel_v_read_readvariableop`savev2_adam_convolutional_autoencoder_1_convolution_decoder_1_dense_3_bias_v_read_readvariableopdsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_bias_v_read_readvariableopdsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_bias_v_read_readvariableopdsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_bias_v_read_readvariableopdsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_bias_v_read_readvariableopdsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?: : : : : : ::: : : @:@:@?:?:	?	::	?:?:?:?:??:?:?@:@:@ : : :: : ::: : : @:@:@?:?:	?	::	?:?:?:?:??:?:?@:@:@ : : :::: : : @:@:@?:?:	?	::	?:?:?:?:??:?:?@:@:@ : : :: 2(
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
:	?	: 

_output_shapes
::%!

_output_shapes
:	?:!
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
:	?	: '

_output_shapes
::%(!

_output_shapes
:	?:!)
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
:	?	: =

_output_shapes
::%>!

_output_shapes
:	?:!?
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
}
(__inference_conv2d_9_layer_call_fn_23515

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
GPU 2J 8? *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_227872
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
?

?
D__inference_conv2d_17_layer_call_and_return_conditional_losses_23201

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
D__inference_conv2d_12_layer_call_and_return_conditional_losses_22868

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
~
)__inference_conv2d_15_layer_call_fn_23635

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
GPU 2J 8? *M
fHRF
D__inference_conv2d_15_layer_call_and_return_conditional_losses_231462
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
?
P__inference_convolution_encoder_1_layer_call_and_return_conditional_losses_22925
input_1
conv2d_9_22798
conv2d_9_22800
conv2d_10_22825
conv2d_10_22827
conv2d_11_22852
conv2d_11_22854
conv2d_12_22879
conv2d_12_22881
dense_2_22919
dense_2_22921
identity??!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_9_22798conv2d_9_22800*
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
GPU 2J 8? *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_227872"
 conv2d_9/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_22825conv2d_10_22827*
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
GPU 2J 8? *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_228142#
!conv2d_10/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_22852conv2d_11_22854*
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
GPU 2J 8? *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_228412#
!conv2d_11/StatefulPartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0conv2d_12_22879conv2d_12_22881*
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
GPU 2J 8? *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_228682#
!conv2d_12/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_228902
flatten_1/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_22919dense_2_22921*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_229082!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????@@::::::::::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_23071

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
?
|
'__inference_dense_3_layer_call_fn_23457

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
B__inference_dense_3_layer_call_and_return_conditional_losses_230412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_conv2d_9_layer_call_and_return_conditional_losses_23506

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
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_23471

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
??
??
!__inference__traced_restore_24126
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rateX
Tassignvariableop_5_convolutional_autoencoder_1_convolution_encoder_1_conv2d_9_kernelV
Rassignvariableop_6_convolutional_autoencoder_1_convolution_encoder_1_conv2d_9_biasY
Uassignvariableop_7_convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_kernelW
Sassignvariableop_8_convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_biasY
Uassignvariableop_9_convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_kernelX
Tassignvariableop_10_convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_biasZ
Vassignvariableop_11_convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_kernelX
Tassignvariableop_12_convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_biasX
Tassignvariableop_13_convolutional_autoencoder_1_convolution_encoder_1_dense_2_kernelV
Rassignvariableop_14_convolutional_autoencoder_1_convolution_encoder_1_dense_2_biasX
Tassignvariableop_15_convolutional_autoencoder_1_convolution_decoder_1_dense_3_kernelV
Rassignvariableop_16_convolutional_autoencoder_1_convolution_decoder_1_dense_3_biasZ
Vassignvariableop_17_convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_kernelX
Tassignvariableop_18_convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_biasZ
Vassignvariableop_19_convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_kernelX
Tassignvariableop_20_convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_biasZ
Vassignvariableop_21_convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_kernelX
Tassignvariableop_22_convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_biasZ
Vassignvariableop_23_convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_kernelX
Tassignvariableop_24_convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_biasZ
Vassignvariableop_25_convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_kernelX
Tassignvariableop_26_convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_bias
assignvariableop_27_total
assignvariableop_28_count`
\assignvariableop_29_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_9_kernel_m^
Zassignvariableop_30_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_9_bias_ma
]assignvariableop_31_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_kernel_m_
[assignvariableop_32_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_bias_ma
]assignvariableop_33_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_kernel_m_
[assignvariableop_34_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_bias_ma
]assignvariableop_35_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_kernel_m_
[assignvariableop_36_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_bias_m_
[assignvariableop_37_adam_convolutional_autoencoder_1_convolution_encoder_1_dense_2_kernel_m]
Yassignvariableop_38_adam_convolutional_autoencoder_1_convolution_encoder_1_dense_2_bias_m_
[assignvariableop_39_adam_convolutional_autoencoder_1_convolution_decoder_1_dense_3_kernel_m]
Yassignvariableop_40_adam_convolutional_autoencoder_1_convolution_decoder_1_dense_3_bias_ma
]assignvariableop_41_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_kernel_m_
[assignvariableop_42_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_bias_ma
]assignvariableop_43_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_kernel_m_
[assignvariableop_44_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_bias_ma
]assignvariableop_45_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_kernel_m_
[assignvariableop_46_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_bias_ma
]assignvariableop_47_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_kernel_m_
[assignvariableop_48_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_bias_ma
]assignvariableop_49_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_kernel_m_
[assignvariableop_50_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_bias_m`
\assignvariableop_51_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_9_kernel_v^
Zassignvariableop_52_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_9_bias_va
]assignvariableop_53_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_kernel_v_
[assignvariableop_54_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_bias_va
]assignvariableop_55_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_kernel_v_
[assignvariableop_56_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_bias_va
]assignvariableop_57_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_kernel_v_
[assignvariableop_58_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_bias_v_
[assignvariableop_59_adam_convolutional_autoencoder_1_convolution_encoder_1_dense_2_kernel_v]
Yassignvariableop_60_adam_convolutional_autoencoder_1_convolution_encoder_1_dense_2_bias_v_
[assignvariableop_61_adam_convolutional_autoencoder_1_convolution_decoder_1_dense_3_kernel_v]
Yassignvariableop_62_adam_convolutional_autoencoder_1_convolution_decoder_1_dense_3_bias_va
]assignvariableop_63_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_kernel_v_
[assignvariableop_64_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_bias_va
]assignvariableop_65_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_kernel_v_
[assignvariableop_66_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_bias_va
]assignvariableop_67_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_kernel_v_
[assignvariableop_68_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_bias_va
]assignvariableop_69_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_kernel_v_
[assignvariableop_70_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_bias_va
]assignvariableop_71_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_kernel_v_
[assignvariableop_72_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_bias_v
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
AssignVariableOp_5AssignVariableOpTassignvariableop_5_convolutional_autoencoder_1_convolution_encoder_1_conv2d_9_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpRassignvariableop_6_convolutional_autoencoder_1_convolution_encoder_1_conv2d_9_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpUassignvariableop_7_convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpSassignvariableop_8_convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpUassignvariableop_9_convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpTassignvariableop_10_convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpVassignvariableop_11_convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpTassignvariableop_12_convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpTassignvariableop_13_convolutional_autoencoder_1_convolution_encoder_1_dense_2_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpRassignvariableop_14_convolutional_autoencoder_1_convolution_encoder_1_dense_2_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpTassignvariableop_15_convolutional_autoencoder_1_convolution_decoder_1_dense_3_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpRassignvariableop_16_convolutional_autoencoder_1_convolution_decoder_1_dense_3_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpVassignvariableop_17_convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpTassignvariableop_18_convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpVassignvariableop_19_convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpTassignvariableop_20_convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpVassignvariableop_21_convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpTassignvariableop_22_convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpVassignvariableop_23_convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpTassignvariableop_24_convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpVassignvariableop_25_convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpTassignvariableop_26_convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_biasIdentity_26:output:0"/device:CPU:0*
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
AssignVariableOp_29AssignVariableOp\assignvariableop_29_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_9_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpZassignvariableop_30_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_9_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp]assignvariableop_31_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp[assignvariableop_32_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp]assignvariableop_33_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp[assignvariableop_34_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp]assignvariableop_35_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp[assignvariableop_36_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp[assignvariableop_37_adam_convolutional_autoencoder_1_convolution_encoder_1_dense_2_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpYassignvariableop_38_adam_convolutional_autoencoder_1_convolution_encoder_1_dense_2_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp[assignvariableop_39_adam_convolutional_autoencoder_1_convolution_decoder_1_dense_3_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpYassignvariableop_40_adam_convolutional_autoencoder_1_convolution_decoder_1_dense_3_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp]assignvariableop_41_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp[assignvariableop_42_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp]assignvariableop_43_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp[assignvariableop_44_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp]assignvariableop_45_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp[assignvariableop_46_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp]assignvariableop_47_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp[assignvariableop_48_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp]assignvariableop_49_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp[assignvariableop_50_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp\assignvariableop_51_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_9_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOpZassignvariableop_52_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_9_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp]assignvariableop_53_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp[assignvariableop_54_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp]assignvariableop_55_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp[assignvariableop_56_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp]assignvariableop_57_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp[assignvariableop_58_adam_convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp[assignvariableop_59_adam_convolutional_autoencoder_1_convolution_encoder_1_dense_2_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOpYassignvariableop_60_adam_convolutional_autoencoder_1_convolution_encoder_1_dense_2_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp[assignvariableop_61_adam_convolutional_autoencoder_1_convolution_decoder_1_dense_3_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOpYassignvariableop_62_adam_convolutional_autoencoder_1_convolution_decoder_1_dense_3_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp]assignvariableop_63_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp[assignvariableop_64_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp]assignvariableop_65_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp[assignvariableop_66_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp]assignvariableop_67_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp[assignvariableop_68_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp]assignvariableop_69_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp[assignvariableop_70_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp]assignvariableop_71_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp[assignvariableop_72_adam_convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_bias_vIdentity_72:output:0"/device:CPU:0*
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
?
f
J__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_22983

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
D__inference_conv2d_15_layer_call_and_return_conditional_losses_23626

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
?
B__inference_dense_3_layer_call_and_return_conditional_losses_23448

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
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
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_15_layer_call_and_return_conditional_losses_23146

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
?
5__inference_convolution_decoder_1_layer_call_fn_23248
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
GPU 2J 8? *Y
fTRR
P__inference_convolution_decoder_1_layer_call_and_return_conditional_losses_232182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
B__inference_dense_2_layer_call_and_return_conditional_losses_22908

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
f
J__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_23002

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
?
E
)__inference_reshape_1_layer_call_fn_23476

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
GPU 2J 8? *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_230712
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
D__inference_conv2d_14_layer_call_and_return_conditional_losses_23118

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
?
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_22890

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
?
~
)__inference_conv2d_13_layer_call_fn_23595

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
GPU 2J 8? *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_230902
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

?
D__inference_conv2d_17_layer_call_and_return_conditional_losses_23486

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
?
K
/__inference_up_sampling2d_7_layer_call_fn_23027

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
J__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_230212
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
?
E
)__inference_flatten_1_layer_call_fn_23419

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
GPU 2J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_228902
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
D__inference_conv2d_16_layer_call_and_return_conditional_losses_23646

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
??
?
 __inference__wrapped_model_22772
input_1]
Yconvolutional_autoencoder_1_convolution_encoder_1_conv2d_9_conv2d_readvariableop_resource^
Zconvolutional_autoencoder_1_convolution_encoder_1_conv2d_9_biasadd_readvariableop_resource^
Zconvolutional_autoencoder_1_convolution_encoder_1_conv2d_10_conv2d_readvariableop_resource_
[convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_biasadd_readvariableop_resource^
Zconvolutional_autoencoder_1_convolution_encoder_1_conv2d_11_conv2d_readvariableop_resource_
[convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_biasadd_readvariableop_resource^
Zconvolutional_autoencoder_1_convolution_encoder_1_conv2d_12_conv2d_readvariableop_resource_
[convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_biasadd_readvariableop_resource\
Xconvolutional_autoencoder_1_convolution_encoder_1_dense_2_matmul_readvariableop_resource]
Yconvolutional_autoencoder_1_convolution_encoder_1_dense_2_biasadd_readvariableop_resource\
Xconvolutional_autoencoder_1_convolution_decoder_1_dense_3_matmul_readvariableop_resource]
Yconvolutional_autoencoder_1_convolution_decoder_1_dense_3_biasadd_readvariableop_resource^
Zconvolutional_autoencoder_1_convolution_decoder_1_conv2d_13_conv2d_readvariableop_resource_
[convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_biasadd_readvariableop_resource^
Zconvolutional_autoencoder_1_convolution_decoder_1_conv2d_14_conv2d_readvariableop_resource_
[convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_biasadd_readvariableop_resource^
Zconvolutional_autoencoder_1_convolution_decoder_1_conv2d_15_conv2d_readvariableop_resource_
[convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_biasadd_readvariableop_resource^
Zconvolutional_autoencoder_1_convolution_decoder_1_conv2d_16_conv2d_readvariableop_resource_
[convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_biasadd_readvariableop_resource^
Zconvolutional_autoencoder_1_convolution_decoder_1_conv2d_17_conv2d_readvariableop_resource_
[convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_biasadd_readvariableop_resource
identity??Rconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/Conv2D/ReadVariableOp?Rconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/Conv2D/ReadVariableOp?Rconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/Conv2D/ReadVariableOp?Rconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/Conv2D/ReadVariableOp?Rconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/Conv2D/ReadVariableOp?Pconvolutional_autoencoder_1/convolution_decoder_1/dense_3/BiasAdd/ReadVariableOp?Oconvolutional_autoencoder_1/convolution_decoder_1/dense_3/MatMul/ReadVariableOp?Rconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/Conv2D/ReadVariableOp?Rconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/Conv2D/ReadVariableOp?Rconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/Conv2D/ReadVariableOp?Qconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/BiasAdd/ReadVariableOp?Pconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/Conv2D/ReadVariableOp?Pconvolutional_autoencoder_1/convolution_encoder_1/dense_2/BiasAdd/ReadVariableOp?Oconvolutional_autoencoder_1/convolution_encoder_1/dense_2/MatMul/ReadVariableOp?
Pconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/Conv2D/ReadVariableOpReadVariableOpYconvolutional_autoencoder_1_convolution_encoder_1_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02R
Pconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/Conv2D/ReadVariableOp?
Aconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/Conv2DConv2Dinput_1Xconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2C
Aconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/Conv2D?
Qconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/BiasAdd/ReadVariableOpReadVariableOpZconvolutional_autoencoder_1_convolution_encoder_1_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02S
Qconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/BiasAdd/ReadVariableOp?
Bconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/BiasAddBiasAddJconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/Conv2D:output:0Yconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2D
Bconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/BiasAdd?
?convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/ReluReluKconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2A
?convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/Relu?
Qconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_1_convolution_encoder_1_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02S
Qconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/Conv2DConv2DMconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/Relu:activations:0Yconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2D
Bconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/Conv2D?
Rconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_1_convolution_encoder_1_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02T
Rconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/BiasAddBiasAddKconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/Conv2D:output:0Zconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2E
Cconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/BiasAdd?
@convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/ReluReluLconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2B
@convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/Relu?
Qconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_1_convolution_encoder_1_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02S
Qconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/Conv2DConv2DNconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/Relu:activations:0Yconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2D
Bconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/Conv2D?
Rconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_1_convolution_encoder_1_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02T
Rconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/BiasAddBiasAddKconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/Conv2D:output:0Zconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2E
Cconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/BiasAdd?
@convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/ReluReluLconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2B
@convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/Relu?
Qconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_1_convolution_encoder_1_conv2d_12_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02S
Qconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/Conv2DConv2DNconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/Relu:activations:0Yconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2D
Bconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/Conv2D?
Rconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_1_convolution_encoder_1_conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02T
Rconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/BiasAddBiasAddKconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/Conv2D:output:0Zconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2E
Cconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/BiasAdd?
@convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/ReluReluLconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2B
@convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/Relu?
Aconvolutional_autoencoder_1/convolution_encoder_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2C
Aconvolutional_autoencoder_1/convolution_encoder_1/flatten_1/Const?
Cconvolutional_autoencoder_1/convolution_encoder_1/flatten_1/ReshapeReshapeNconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/Relu:activations:0Jconvolutional_autoencoder_1/convolution_encoder_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????	2E
Cconvolutional_autoencoder_1/convolution_encoder_1/flatten_1/Reshape?
Oconvolutional_autoencoder_1/convolution_encoder_1/dense_2/MatMul/ReadVariableOpReadVariableOpXconvolutional_autoencoder_1_convolution_encoder_1_dense_2_matmul_readvariableop_resource*
_output_shapes
:	?	*
dtype02Q
Oconvolutional_autoencoder_1/convolution_encoder_1/dense_2/MatMul/ReadVariableOp?
@convolutional_autoencoder_1/convolution_encoder_1/dense_2/MatMulMatMulLconvolutional_autoencoder_1/convolution_encoder_1/flatten_1/Reshape:output:0Wconvolutional_autoencoder_1/convolution_encoder_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2B
@convolutional_autoencoder_1/convolution_encoder_1/dense_2/MatMul?
Pconvolutional_autoencoder_1/convolution_encoder_1/dense_2/BiasAdd/ReadVariableOpReadVariableOpYconvolutional_autoencoder_1_convolution_encoder_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02R
Pconvolutional_autoencoder_1/convolution_encoder_1/dense_2/BiasAdd/ReadVariableOp?
Aconvolutional_autoencoder_1/convolution_encoder_1/dense_2/BiasAddBiasAddJconvolutional_autoencoder_1/convolution_encoder_1/dense_2/MatMul:product:0Xconvolutional_autoencoder_1/convolution_encoder_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2C
Aconvolutional_autoencoder_1/convolution_encoder_1/dense_2/BiasAdd?
Oconvolutional_autoencoder_1/convolution_decoder_1/dense_3/MatMul/ReadVariableOpReadVariableOpXconvolutional_autoencoder_1_convolution_decoder_1_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02Q
Oconvolutional_autoencoder_1/convolution_decoder_1/dense_3/MatMul/ReadVariableOp?
@convolutional_autoencoder_1/convolution_decoder_1/dense_3/MatMulMatMulJconvolutional_autoencoder_1/convolution_encoder_1/dense_2/BiasAdd:output:0Wconvolutional_autoencoder_1/convolution_decoder_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2B
@convolutional_autoencoder_1/convolution_decoder_1/dense_3/MatMul?
Pconvolutional_autoencoder_1/convolution_decoder_1/dense_3/BiasAdd/ReadVariableOpReadVariableOpYconvolutional_autoencoder_1_convolution_decoder_1_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02R
Pconvolutional_autoencoder_1/convolution_decoder_1/dense_3/BiasAdd/ReadVariableOp?
Aconvolutional_autoencoder_1/convolution_decoder_1/dense_3/BiasAddBiasAddJconvolutional_autoencoder_1/convolution_decoder_1/dense_3/MatMul:product:0Xconvolutional_autoencoder_1/convolution_decoder_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2C
Aconvolutional_autoencoder_1/convolution_decoder_1/dense_3/BiasAdd?
Aconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/ShapeShapeJconvolutional_autoencoder_1/convolution_decoder_1/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:2C
Aconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/Shape?
Oconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Q
Oconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/strided_slice/stack?
Qconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2S
Qconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/strided_slice/stack_1?
Qconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
Qconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/strided_slice/stack_2?
Iconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/strided_sliceStridedSliceJconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/Shape:output:0Xconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/strided_slice/stack:output:0Zconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/strided_slice/stack_1:output:0Zconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2K
Iconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/strided_slice?
Kconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2M
Kconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/Reshape/shape/1?
Kconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2M
Kconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/Reshape/shape/2?
Kconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2M
Kconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/Reshape/shape/3?
Iconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/Reshape/shapePackRconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/strided_slice:output:0Tconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/Reshape/shape/1:output:0Tconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/Reshape/shape/2:output:0Tconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2K
Iconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/Reshape/shape?
Cconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/ReshapeReshapeJconvolutional_autoencoder_1/convolution_decoder_1/dense_3/BiasAdd:output:0Rconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2E
Cconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/Reshape?
Qconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_1_convolution_decoder_1_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02S
Qconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/Conv2DConv2DLconvolutional_autoencoder_1/convolution_decoder_1/reshape_1/Reshape:output:0Yconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2D
Bconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/Conv2D?
Rconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_1_convolution_decoder_1_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02T
Rconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/BiasAddBiasAddKconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/Conv2D:output:0Zconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2E
Cconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/BiasAdd?
@convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/ReluReluLconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2B
@convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/Relu?
Gconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_4/ShapeShapeNconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/Relu:activations:0*
T0*
_output_shapes
:2I
Gconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_4/Shape?
Uconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2W
Uconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_4/strided_slice/stack?
Wconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Y
Wconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_4/strided_slice/stack_1?
Wconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Y
Wconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_4/strided_slice/stack_2?
Oconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_4/strided_sliceStridedSlicePconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_4/Shape:output:0^convolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_4/strided_slice/stack:output:0`convolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_4/strided_slice/stack_1:output:0`convolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2Q
Oconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_4/strided_slice?
Gconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2I
Gconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_4/Const?
Econvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_4/mulMulXconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_4/strided_slice:output:0Pconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2G
Econvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_4/mul?
^convolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighborNconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/Relu:activations:0Iconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_4/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2`
^convolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_4/resize/ResizeNearestNeighbor?
Qconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_1_convolution_decoder_1_conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02S
Qconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/Conv2DConv2Doconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0Yconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2D
Bconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/Conv2D?
Rconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_1_convolution_decoder_1_conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02T
Rconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/BiasAddBiasAddKconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/Conv2D:output:0Zconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2E
Cconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/BiasAdd?
@convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/ReluReluLconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2B
@convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/Relu?
Gconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_5/ShapeShapeNconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/Relu:activations:0*
T0*
_output_shapes
:2I
Gconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_5/Shape?
Uconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2W
Uconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_5/strided_slice/stack?
Wconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Y
Wconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_5/strided_slice/stack_1?
Wconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Y
Wconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_5/strided_slice/stack_2?
Oconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_5/strided_sliceStridedSlicePconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_5/Shape:output:0^convolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_5/strided_slice/stack:output:0`convolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_5/strided_slice/stack_1:output:0`convolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2Q
Oconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_5/strided_slice?
Gconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2I
Gconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_5/Const?
Econvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_5/mulMulXconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_5/strided_slice:output:0Pconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_5/Const:output:0*
T0*
_output_shapes
:2G
Econvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_5/mul?
^convolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_5/resize/ResizeNearestNeighborResizeNearestNeighborNconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/Relu:activations:0Iconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_5/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2`
^convolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_5/resize/ResizeNearestNeighbor?
Qconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_1_convolution_decoder_1_conv2d_15_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02S
Qconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/Conv2DConv2Doconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_5/resize/ResizeNearestNeighbor:resized_images:0Yconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2D
Bconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/Conv2D?
Rconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_1_convolution_decoder_1_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02T
Rconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/BiasAddBiasAddKconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/Conv2D:output:0Zconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2E
Cconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/BiasAdd?
@convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/ReluReluLconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2B
@convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/Relu?
Gconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_6/ShapeShapeNconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/Relu:activations:0*
T0*
_output_shapes
:2I
Gconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_6/Shape?
Uconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2W
Uconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_6/strided_slice/stack?
Wconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Y
Wconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_6/strided_slice/stack_1?
Wconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Y
Wconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_6/strided_slice/stack_2?
Oconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_6/strided_sliceStridedSlicePconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_6/Shape:output:0^convolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_6/strided_slice/stack:output:0`convolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_6/strided_slice/stack_1:output:0`convolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2Q
Oconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_6/strided_slice?
Gconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2I
Gconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_6/Const?
Econvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_6/mulMulXconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_6/strided_slice:output:0Pconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_6/Const:output:0*
T0*
_output_shapes
:2G
Econvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_6/mul?
^convolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighborNconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/Relu:activations:0Iconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_6/mul:z:0*
T0*/
_output_shapes
:?????????  @*
half_pixel_centers(2`
^convolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_6/resize/ResizeNearestNeighbor?
Qconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_1_convolution_decoder_1_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02S
Qconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/Conv2DConv2Doconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0Yconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2D
Bconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/Conv2D?
Rconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_1_convolution_decoder_1_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02T
Rconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/BiasAddBiasAddKconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/Conv2D:output:0Zconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2E
Cconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/BiasAdd?
@convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/ReluReluLconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2B
@convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/Relu?
Gconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_7/ShapeShapeNconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/Relu:activations:0*
T0*
_output_shapes
:2I
Gconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_7/Shape?
Uconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2W
Uconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_7/strided_slice/stack?
Wconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Y
Wconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_7/strided_slice/stack_1?
Wconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Y
Wconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_7/strided_slice/stack_2?
Oconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_7/strided_sliceStridedSlicePconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_7/Shape:output:0^convolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_7/strided_slice/stack:output:0`convolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_7/strided_slice/stack_1:output:0`convolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2Q
Oconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_7/strided_slice?
Gconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2I
Gconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_7/Const?
Econvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_7/mulMulXconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_7/strided_slice:output:0Pconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_7/Const:output:0*
T0*
_output_shapes
:2G
Econvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_7/mul?
^convolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighborNconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/Relu:activations:0Iconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_7/mul:z:0*
T0*/
_output_shapes
:?????????@@ *
half_pixel_centers(2`
^convolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_7/resize/ResizeNearestNeighbor?
Qconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_1_convolution_decoder_1_conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02S
Qconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/Conv2DConv2Doconvolutional_autoencoder_1/convolution_decoder_1/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0Yconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2D
Bconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/Conv2D?
Rconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_1_convolution_decoder_1_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02T
Rconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/BiasAddBiasAddKconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/Conv2D:output:0Zconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2E
Cconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/BiasAdd?
IdentityIdentityLconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/BiasAdd:output:0S^convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/BiasAdd/ReadVariableOpR^convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/Conv2D/ReadVariableOpS^convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/BiasAdd/ReadVariableOpR^convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/Conv2D/ReadVariableOpS^convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/BiasAdd/ReadVariableOpR^convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/Conv2D/ReadVariableOpS^convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/BiasAdd/ReadVariableOpR^convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/Conv2D/ReadVariableOpS^convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/BiasAdd/ReadVariableOpR^convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/Conv2D/ReadVariableOpQ^convolutional_autoencoder_1/convolution_decoder_1/dense_3/BiasAdd/ReadVariableOpP^convolutional_autoencoder_1/convolution_decoder_1/dense_3/MatMul/ReadVariableOpS^convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/BiasAdd/ReadVariableOpR^convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/Conv2D/ReadVariableOpS^convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/BiasAdd/ReadVariableOpR^convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/Conv2D/ReadVariableOpS^convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/BiasAdd/ReadVariableOpR^convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/Conv2D/ReadVariableOpR^convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/BiasAdd/ReadVariableOpQ^convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/Conv2D/ReadVariableOpQ^convolutional_autoencoder_1/convolution_encoder_1/dense_2/BiasAdd/ReadVariableOpP^convolutional_autoencoder_1/convolution_encoder_1/dense_2/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????@@::::::::::::::::::::::2?
Rconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/BiasAdd/ReadVariableOpRconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/Conv2D/ReadVariableOpQconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/Conv2D/ReadVariableOp2?
Rconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/BiasAdd/ReadVariableOpRconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/Conv2D/ReadVariableOpQconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/Conv2D/ReadVariableOp2?
Rconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/BiasAdd/ReadVariableOpRconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/Conv2D/ReadVariableOpQconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/Conv2D/ReadVariableOp2?
Rconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/BiasAdd/ReadVariableOpRconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/Conv2D/ReadVariableOpQconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/Conv2D/ReadVariableOp2?
Rconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/BiasAdd/ReadVariableOpRconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/Conv2D/ReadVariableOpQconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/Conv2D/ReadVariableOp2?
Pconvolutional_autoencoder_1/convolution_decoder_1/dense_3/BiasAdd/ReadVariableOpPconvolutional_autoencoder_1/convolution_decoder_1/dense_3/BiasAdd/ReadVariableOp2?
Oconvolutional_autoencoder_1/convolution_decoder_1/dense_3/MatMul/ReadVariableOpOconvolutional_autoencoder_1/convolution_decoder_1/dense_3/MatMul/ReadVariableOp2?
Rconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/BiasAdd/ReadVariableOpRconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/Conv2D/ReadVariableOpQconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/Conv2D/ReadVariableOp2?
Rconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/BiasAdd/ReadVariableOpRconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/Conv2D/ReadVariableOpQconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/Conv2D/ReadVariableOp2?
Rconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/BiasAdd/ReadVariableOpRconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/Conv2D/ReadVariableOpQconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/Conv2D/ReadVariableOp2?
Qconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/BiasAdd/ReadVariableOpQconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/BiasAdd/ReadVariableOp2?
Pconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/Conv2D/ReadVariableOpPconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/Conv2D/ReadVariableOp2?
Pconvolutional_autoencoder_1/convolution_encoder_1/dense_2/BiasAdd/ReadVariableOpPconvolutional_autoencoder_1/convolution_encoder_1/dense_2/BiasAdd/ReadVariableOp2?
Oconvolutional_autoencoder_1/convolution_encoder_1/dense_2/MatMul/ReadVariableOpOconvolutional_autoencoder_1/convolution_encoder_1/dense_2/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?

?
D__inference_conv2d_13_layer_call_and_return_conditional_losses_23090

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
?
K
/__inference_up_sampling2d_4_layer_call_fn_22970

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
J__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_229642
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
D__inference_conv2d_11_layer_call_and_return_conditional_losses_22841

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
?
~
)__inference_conv2d_10_layer_call_fn_23535

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
GPU 2J 8? *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_228142
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
?
~
)__inference_conv2d_17_layer_call_fn_23495

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
GPU 2J 8? *M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_232012
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
J__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_22964

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
?1
?
P__inference_convolution_decoder_1_layer_call_and_return_conditional_losses_23218
input_1
dense_3_23052
dense_3_23054
conv2d_13_23101
conv2d_13_23103
conv2d_14_23129
conv2d_14_23131
conv2d_15_23157
conv2d_15_23159
conv2d_16_23185
conv2d_16_23187
conv2d_17_23212
conv2d_17_23214
identity??!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_3_23052dense_3_23054*
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
B__inference_dense_3_layer_call_and_return_conditional_losses_230412!
dense_3/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_230712
reshape_1/PartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_13_23101conv2d_13_23103*
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
GPU 2J 8? *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_230902#
!conv2d_13/StatefulPartitionedCall?
up_sampling2d_4/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *S
fNRL
J__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_229642!
up_sampling2d_4/PartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_4/PartitionedCall:output:0conv2d_14_23129conv2d_14_23131*
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
GPU 2J 8? *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_231182#
!conv2d_14/StatefulPartitionedCall?
up_sampling2d_5/PartitionedCallPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0*
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
J__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_229832!
up_sampling2d_5/PartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_5/PartitionedCall:output:0conv2d_15_23157conv2d_15_23159*
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
GPU 2J 8? *M
fHRF
D__inference_conv2d_15_layer_call_and_return_conditional_losses_231462#
!conv2d_15/StatefulPartitionedCall?
up_sampling2d_6/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
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
J__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_230022!
up_sampling2d_6/PartitionedCall?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_6/PartitionedCall:output:0conv2d_16_23185conv2d_16_23187*
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
GPU 2J 8? *M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_231742#
!conv2d_16/StatefulPartitionedCall?
up_sampling2d_7/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
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
J__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_230212!
up_sampling2d_7/PartitionedCall?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_7/PartitionedCall:output:0conv2d_17_23212conv2d_17_23214*
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
GPU 2J 8? *M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_232012#
!conv2d_17/StatefulPartitionedCall?
IdentityIdentity*conv2d_17/StatefulPartitionedCall:output:0"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::::2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
D__inference_conv2d_11_layer_call_and_return_conditional_losses_23546

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
?
~
)__inference_conv2d_14_layer_call_fn_23615

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
GPU 2J 8? *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_231182
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
?
#__inference_signature_wrapper_23408
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
 __inference__wrapped_model_227722
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
|
'__inference_dense_2_layer_call_fn_23438

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
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_229082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?	
?
B__inference_dense_2_layer_call_and_return_conditional_losses_23429

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
?
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_23414

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
D__inference_conv2d_10_layer_call_and_return_conditional_losses_23526

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
?
~
)__inference_conv2d_11_layer_call_fn_23555

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
GPU 2J 8? *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_228412
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
?
B__inference_dense_3_layer_call_and_return_conditional_losses_23041

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
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
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_16_layer_call_and_return_conditional_losses_23174

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
K
/__inference_up_sampling2d_6_layer_call_fn_23008

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
J__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_230022
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
?
~
)__inference_conv2d_12_layer_call_fn_23575

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
GPU 2J 8? *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_228682
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

?
C__inference_conv2d_9_layer_call_and_return_conditional_losses_22787

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

?
D__inference_conv2d_12_layer_call_and_return_conditional_losses_23566

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
?
K
/__inference_up_sampling2d_5_layer_call_fn_22989

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
J__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_229832
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
?
;__inference_convolutional_autoencoder_1_layer_call_fn_23349
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
GPU 2J 8? *_
fZRX
V__inference_convolutional_autoencoder_1_layer_call_and_return_conditional_losses_232992
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
_user_specified_name	input_1"?L
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
_tf_keras_model?{"class_name": "ConvolutionalAutoencoder", "name": "convolutional_autoencoder_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"image_size": 64, "code_dim": 16, "depth": 4}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 64, 64, 1]}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ConvolutionalAutoencoder", "config": {"image_size": 64, "code_dim": 16, "depth": 4}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
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
_tf_keras_model?{"class_name": "ConvolutionEncoder", "name": "convolution_encoder_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 64, 64, 1]}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ConvolutionEncoder"}}
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
_tf_keras_model?{"class_name": "ConvolutionDecoder", "name": "convolution_decoder_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 16]}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ConvolutionDecoder"}}
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
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

&kernel
'bias
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1152}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 1152]}}
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
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 16]}}
?
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4, 16]}}}
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
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 64, 64, 32]}}
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
[:Y2Aconvolutional_autoencoder_1/convolution_encoder_1/conv2d_9/kernel
M:K2?convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/bias
\:Z 2Bconvolutional_autoencoder_1/convolution_encoder_1/conv2d_10/kernel
N:L 2@convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/bias
\:Z @2Bconvolutional_autoencoder_1/convolution_encoder_1/conv2d_11/kernel
N:L@2@convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/bias
]:[@?2Bconvolutional_autoencoder_1/convolution_encoder_1/conv2d_12/kernel
O:M?2@convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/bias
S:Q	?	2@convolutional_autoencoder_1/convolution_encoder_1/dense_2/kernel
L:J2>convolutional_autoencoder_1/convolution_encoder_1/dense_2/bias
S:Q	?2@convolutional_autoencoder_1/convolution_decoder_1/dense_3/kernel
M:K?2>convolutional_autoencoder_1/convolution_decoder_1/dense_3/bias
]:[?2Bconvolutional_autoencoder_1/convolution_decoder_1/conv2d_13/kernel
O:M?2@convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/bias
^:\??2Bconvolutional_autoencoder_1/convolution_decoder_1/conv2d_14/kernel
O:M?2@convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/bias
]:[?@2Bconvolutional_autoencoder_1/convolution_decoder_1/conv2d_15/kernel
N:L@2@convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/bias
\:Z@ 2Bconvolutional_autoencoder_1/convolution_decoder_1/conv2d_16/kernel
N:L 2@convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/bias
\:Z 2Bconvolutional_autoencoder_1/convolution_decoder_1/conv2d_17/kernel
N:L2@convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/bias
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
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 64, 64, 1]}}
?	

 kernel
!bias
hregularization_losses
itrainable_variables
j	variables
k	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 31, 31, 16]}}
?	

"kernel
#bias
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 15, 15, 32]}}
?	

$kernel
%bias
pregularization_losses
qtrainable_variables
r	variables
s	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 7, 7, 64]}}
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
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 4, 4, 16]}}
?	

,kernel
-bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 8, 8, 256]}}
?	

.kernel
/bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 16, 16, 128]}}
?	

0kernel
1bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 32, 32, 64]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_5", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_6", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_7", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
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
`:^2HAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/kernel/m
R:P2FAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/bias/m
a:_ 2IAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/kernel/m
S:Q 2GAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/bias/m
a:_ @2IAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/kernel/m
S:Q@2GAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/bias/m
b:`@?2IAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/kernel/m
T:R?2GAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/bias/m
X:V	?	2GAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/kernel/m
Q:O2EAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/bias/m
X:V	?2GAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/kernel/m
R:P?2EAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/bias/m
b:`?2IAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/kernel/m
T:R?2GAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/bias/m
c:a??2IAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/kernel/m
T:R?2GAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/bias/m
b:`?@2IAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/kernel/m
S:Q@2GAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/bias/m
a:_@ 2IAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/kernel/m
S:Q 2GAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/bias/m
a:_ 2IAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/kernel/m
S:Q2GAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/bias/m
`:^2HAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/kernel/v
R:P2FAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_9/bias/v
a:_ 2IAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/kernel/v
S:Q 2GAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_10/bias/v
a:_ @2IAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/kernel/v
S:Q@2GAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_11/bias/v
b:`@?2IAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/kernel/v
T:R?2GAdam/convolutional_autoencoder_1/convolution_encoder_1/conv2d_12/bias/v
X:V	?	2GAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/kernel/v
Q:O2EAdam/convolutional_autoencoder_1/convolution_encoder_1/dense_2/bias/v
X:V	?2GAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/kernel/v
R:P?2EAdam/convolutional_autoencoder_1/convolution_decoder_1/dense_3/bias/v
b:`?2IAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/kernel/v
T:R?2GAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_13/bias/v
c:a??2IAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/kernel/v
T:R?2GAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_14/bias/v
b:`?@2IAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/kernel/v
S:Q@2GAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_15/bias/v
a:_@ 2IAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/kernel/v
S:Q 2GAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_16/bias/v
a:_ 2IAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/kernel/v
S:Q2GAdam/convolutional_autoencoder_1/convolution_decoder_1/conv2d_17/bias/v
?2?
;__inference_convolutional_autoencoder_1_layer_call_fn_23349?
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
 __inference__wrapped_model_22772?
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
V__inference_convolutional_autoencoder_1_layer_call_and_return_conditional_losses_23299?
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
5__inference_convolution_encoder_1_layer_call_fn_22951?
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
P__inference_convolution_encoder_1_layer_call_and_return_conditional_losses_22925?
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
5__inference_convolution_decoder_1_layer_call_fn_23248?
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
input_1?????????
?2?
P__inference_convolution_decoder_1_layer_call_and_return_conditional_losses_23218?
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
input_1?????????
?B?
#__inference_signature_wrapper_23408input_1"?
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
)__inference_flatten_1_layer_call_fn_23419?
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
D__inference_flatten_1_layer_call_and_return_conditional_losses_23414?
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
'__inference_dense_2_layer_call_fn_23438?
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
B__inference_dense_2_layer_call_and_return_conditional_losses_23429?
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
'__inference_dense_3_layer_call_fn_23457?
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
B__inference_dense_3_layer_call_and_return_conditional_losses_23448?
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
)__inference_reshape_1_layer_call_fn_23476?
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
D__inference_reshape_1_layer_call_and_return_conditional_losses_23471?
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
)__inference_conv2d_17_layer_call_fn_23495?
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
D__inference_conv2d_17_layer_call_and_return_conditional_losses_23486?
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
(__inference_conv2d_9_layer_call_fn_23515?
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
C__inference_conv2d_9_layer_call_and_return_conditional_losses_23506?
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
)__inference_conv2d_10_layer_call_fn_23535?
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
D__inference_conv2d_10_layer_call_and_return_conditional_losses_23526?
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
)__inference_conv2d_11_layer_call_fn_23555?
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
D__inference_conv2d_11_layer_call_and_return_conditional_losses_23546?
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
)__inference_conv2d_12_layer_call_fn_23575?
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
D__inference_conv2d_12_layer_call_and_return_conditional_losses_23566?
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
)__inference_conv2d_13_layer_call_fn_23595?
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
D__inference_conv2d_13_layer_call_and_return_conditional_losses_23586?
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
)__inference_conv2d_14_layer_call_fn_23615?
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
D__inference_conv2d_14_layer_call_and_return_conditional_losses_23606?
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
)__inference_conv2d_15_layer_call_fn_23635?
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
D__inference_conv2d_15_layer_call_and_return_conditional_losses_23626?
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
)__inference_conv2d_16_layer_call_fn_23655?
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
D__inference_conv2d_16_layer_call_and_return_conditional_losses_23646?
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
/__inference_up_sampling2d_4_layer_call_fn_22970?
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
J__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_22964?
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
/__inference_up_sampling2d_5_layer_call_fn_22989?
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
J__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_22983?
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
/__inference_up_sampling2d_6_layer_call_fn_23008?
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
J__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_23002?
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
/__inference_up_sampling2d_7_layer_call_fn_23027?
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
J__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_23021?
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
 __inference__wrapped_model_22772? !"#$%&'()*+,-./01238?5
.?+
)?&
input_1?????????@@
? ";?8
6
output_1*?'
output_1?????????@@?
D__inference_conv2d_10_layer_call_and_return_conditional_losses_23526l !7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
)__inference_conv2d_10_layer_call_fn_23535_ !7?4
-?*
(?%
inputs?????????
? " ?????????? ?
D__inference_conv2d_11_layer_call_and_return_conditional_losses_23546l"#7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
)__inference_conv2d_11_layer_call_fn_23555_"#7?4
-?*
(?%
inputs????????? 
? " ??????????@?
D__inference_conv2d_12_layer_call_and_return_conditional_losses_23566m$%7?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0??????????
? ?
)__inference_conv2d_12_layer_call_fn_23575`$%7?4
-?*
(?%
inputs?????????@
? "!????????????
D__inference_conv2d_13_layer_call_and_return_conditional_losses_23586m*+7?4
-?*
(?%
inputs?????????
? ".?+
$?!
0??????????
? ?
)__inference_conv2d_13_layer_call_fn_23595`*+7?4
-?*
(?%
inputs?????????
? "!????????????
D__inference_conv2d_14_layer_call_and_return_conditional_losses_23606?,-J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
)__inference_conv2d_14_layer_call_fn_23615?,-J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
D__inference_conv2d_15_layer_call_and_return_conditional_losses_23626?./J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
)__inference_conv2d_15_layer_call_fn_23635?./J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+???????????????????????????@?
D__inference_conv2d_16_layer_call_and_return_conditional_losses_23646?01I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
)__inference_conv2d_16_layer_call_fn_23655?01I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
D__inference_conv2d_17_layer_call_and_return_conditional_losses_23486?23I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
)__inference_conv2d_17_layer_call_fn_23495?23I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
C__inference_conv2d_9_layer_call_and_return_conditional_losses_23506l7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????
? ?
(__inference_conv2d_9_layer_call_fn_23515_7?4
-?*
(?%
inputs?????????@@
? " ???????????
P__inference_convolution_decoder_1_layer_call_and_return_conditional_losses_23218?()*+,-./01230?-
&?#
!?
input_1?????????
? "??<
5?2
0+???????????????????????????
? ?
5__inference_convolution_decoder_1_layer_call_fn_23248t()*+,-./01230?-
&?#
!?
input_1?????????
? "2?/+????????????????????????????
P__inference_convolution_encoder_1_layer_call_and_return_conditional_losses_22925m
 !"#$%&'8?5
.?+
)?&
input_1?????????@@
? "%?"
?
0?????????
? ?
5__inference_convolution_encoder_1_layer_call_fn_22951`
 !"#$%&'8?5
.?+
)?&
input_1?????????@@
? "???????????
V__inference_convolutional_autoencoder_1_layer_call_and_return_conditional_losses_23299? !"#$%&'()*+,-./01238?5
.?+
)?&
input_1?????????@@
? "??<
5?2
0+???????????????????????????
? ?
;__inference_convolutional_autoencoder_1_layer_call_fn_23349? !"#$%&'()*+,-./01238?5
.?+
)?&
input_1?????????@@
? "2?/+????????????????????????????
B__inference_dense_2_layer_call_and_return_conditional_losses_23429]&'0?-
&?#
!?
inputs??????????	
? "%?"
?
0?????????
? {
'__inference_dense_2_layer_call_fn_23438P&'0?-
&?#
!?
inputs??????????	
? "???????????
B__inference_dense_3_layer_call_and_return_conditional_losses_23448]()/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? {
'__inference_dense_3_layer_call_fn_23457P()/?,
%?"
 ?
inputs?????????
? "????????????
D__inference_flatten_1_layer_call_and_return_conditional_losses_23414b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????	
? ?
)__inference_flatten_1_layer_call_fn_23419U8?5
.?+
)?&
inputs??????????
? "???????????	?
D__inference_reshape_1_layer_call_and_return_conditional_losses_23471a0?-
&?#
!?
inputs??????????
? "-?*
#? 
0?????????
? ?
)__inference_reshape_1_layer_call_fn_23476T0?-
&?#
!?
inputs??????????
? " ???????????
#__inference_signature_wrapper_23408? !"#$%&'()*+,-./0123C?@
? 
9?6
4
input_1)?&
input_1?????????@@";?8
6
output_1*?'
output_1?????????@@?
J__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_22964?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_up_sampling2d_4_layer_call_fn_22970?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_22983?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_up_sampling2d_5_layer_call_fn_22989?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_23002?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_up_sampling2d_6_layer_call_fn_23008?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_23021?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_up_sampling2d_7_layer_call_fn_23027?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????