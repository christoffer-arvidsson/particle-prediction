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
Bconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*S
shared_nameDBconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/kernel
?
Vconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/kernel*&
_output_shapes
:*
dtype0
?
@convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/bias
?
Tconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/bias*
_output_shapes
:*
dtype0
?
Bconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *S
shared_nameDBconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/kernel
?
Vconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/kernel*&
_output_shapes
: *
dtype0
?
@convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/bias
?
Tconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/bias*
_output_shapes
: *
dtype0
?
Bconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*S
shared_nameDBconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/kernel
?
Vconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/kernel*&
_output_shapes
: @*
dtype0
?
@convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*Q
shared_nameB@convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/bias
?
Tconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/bias*
_output_shapes
:@*
dtype0
?
Bconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*S
shared_nameDBconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/kernel
?
Vconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/kernel*'
_output_shapes
:@?*
dtype0
?
@convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*Q
shared_nameB@convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/bias
?
Tconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/bias*
_output_shapes	
:?*
dtype0
?
@convolutional_autoencoder_4/convolution_encoder_4/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?	?*Q
shared_nameB@convolutional_autoencoder_4/convolution_encoder_4/dense_8/kernel
?
Tconvolutional_autoencoder_4/convolution_encoder_4/dense_8/kernel/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_4/convolution_encoder_4/dense_8/kernel* 
_output_shapes
:
?	?*
dtype0
?
>convolutional_autoencoder_4/convolution_encoder_4/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*O
shared_name@>convolutional_autoencoder_4/convolution_encoder_4/dense_8/bias
?
Rconvolutional_autoencoder_4/convolution_encoder_4/dense_8/bias/Read/ReadVariableOpReadVariableOp>convolutional_autoencoder_4/convolution_encoder_4/dense_8/bias*
_output_shapes	
:?*
dtype0
?
@convolutional_autoencoder_4/convolution_decoder_4/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*Q
shared_nameB@convolutional_autoencoder_4/convolution_decoder_4/dense_9/kernel
?
Tconvolutional_autoencoder_4/convolution_decoder_4/dense_9/kernel/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_4/convolution_decoder_4/dense_9/kernel* 
_output_shapes
:
??*
dtype0
?
>convolutional_autoencoder_4/convolution_decoder_4/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*O
shared_name@>convolutional_autoencoder_4/convolution_decoder_4/dense_9/bias
?
Rconvolutional_autoencoder_4/convolution_decoder_4/dense_9/bias/Read/ReadVariableOpReadVariableOp>convolutional_autoencoder_4/convolution_decoder_4/dense_9/bias*
_output_shapes	
:?*
dtype0
?
Bconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*S
shared_nameDBconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/kernel
?
Vconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/kernel*'
_output_shapes
:?*
dtype0
?
@convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*Q
shared_nameB@convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/bias
?
Tconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/bias*
_output_shapes	
:?*
dtype0
?
Bconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*S
shared_nameDBconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/kernel
?
Vconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/kernel*(
_output_shapes
:??*
dtype0
?
@convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*Q
shared_nameB@convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/bias
?
Tconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/bias*
_output_shapes	
:?*
dtype0
?
Bconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*S
shared_nameDBconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/kernel
?
Vconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/kernel*'
_output_shapes
:?@*
dtype0
?
@convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*Q
shared_nameB@convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/bias
?
Tconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/bias*
_output_shapes
:@*
dtype0
?
Bconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *S
shared_nameDBconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/kernel
?
Vconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/kernel*&
_output_shapes
:@ *
dtype0
?
@convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/bias
?
Tconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/bias*
_output_shapes
: *
dtype0
?
Bconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *S
shared_nameDBconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/kernel
?
Vconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/kernel*&
_output_shapes
: *
dtype0
?
@convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/bias
?
Tconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/bias*
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
IAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Z
shared_nameKIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/kernel/m
?
]Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/kernel/m*&
_output_shapes
:*
dtype0
?
GAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*X
shared_nameIGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/bias/m
?
[Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/bias/m*
_output_shapes
:*
dtype0
?
IAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Z
shared_nameKIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/kernel/m
?
]Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/kernel/m*&
_output_shapes
: *
dtype0
?
GAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/bias/m
?
[Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/bias/m*
_output_shapes
: *
dtype0
?
IAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*Z
shared_nameKIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/kernel/m
?
]Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/kernel/m*&
_output_shapes
: @*
dtype0
?
GAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*X
shared_nameIGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/bias/m
?
[Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/bias/m*
_output_shapes
:@*
dtype0
?
IAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*Z
shared_nameKIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/kernel/m
?
]Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/kernel/m*'
_output_shapes
:@?*
dtype0
?
GAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*X
shared_nameIGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/bias/m
?
[Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/bias/m*
_output_shapes	
:?*
dtype0
?
GAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?	?*X
shared_nameIGAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/kernel/m
?
[Adam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/kernel/m* 
_output_shapes
:
?	?*
dtype0
?
EAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*V
shared_nameGEAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/bias/m
?
YAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/bias/m/Read/ReadVariableOpReadVariableOpEAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/bias/m*
_output_shapes	
:?*
dtype0
?
GAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*X
shared_nameIGAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/kernel/m
?
[Adam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/kernel/m* 
_output_shapes
:
??*
dtype0
?
EAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*V
shared_nameGEAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/bias/m
?
YAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/bias/m/Read/ReadVariableOpReadVariableOpEAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/bias/m*
_output_shapes	
:?*
dtype0
?
IAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*Z
shared_nameKIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/kernel/m
?
]Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/kernel/m*'
_output_shapes
:?*
dtype0
?
GAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*X
shared_nameIGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/bias/m
?
[Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/bias/m*
_output_shapes	
:?*
dtype0
?
IAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*Z
shared_nameKIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/kernel/m
?
]Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/kernel/m*(
_output_shapes
:??*
dtype0
?
GAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*X
shared_nameIGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/bias/m
?
[Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/bias/m*
_output_shapes	
:?*
dtype0
?
IAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*Z
shared_nameKIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/kernel/m
?
]Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/kernel/m*'
_output_shapes
:?@*
dtype0
?
GAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*X
shared_nameIGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/bias/m
?
[Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/bias/m*
_output_shapes
:@*
dtype0
?
IAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *Z
shared_nameKIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/kernel/m
?
]Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/kernel/m*&
_output_shapes
:@ *
dtype0
?
GAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/bias/m
?
[Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/bias/m*
_output_shapes
: *
dtype0
?
IAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Z
shared_nameKIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/kernel/m
?
]Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/kernel/m*&
_output_shapes
: *
dtype0
?
GAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*X
shared_nameIGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/bias/m
?
[Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/bias/m*
_output_shapes
:*
dtype0
?
IAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Z
shared_nameKIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/kernel/v
?
]Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/kernel/v*&
_output_shapes
:*
dtype0
?
GAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*X
shared_nameIGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/bias/v
?
[Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/bias/v*
_output_shapes
:*
dtype0
?
IAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Z
shared_nameKIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/kernel/v
?
]Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/kernel/v*&
_output_shapes
: *
dtype0
?
GAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/bias/v
?
[Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/bias/v*
_output_shapes
: *
dtype0
?
IAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*Z
shared_nameKIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/kernel/v
?
]Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/kernel/v*&
_output_shapes
: @*
dtype0
?
GAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*X
shared_nameIGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/bias/v
?
[Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/bias/v*
_output_shapes
:@*
dtype0
?
IAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*Z
shared_nameKIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/kernel/v
?
]Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/kernel/v*'
_output_shapes
:@?*
dtype0
?
GAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*X
shared_nameIGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/bias/v
?
[Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/bias/v*
_output_shapes	
:?*
dtype0
?
GAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?	?*X
shared_nameIGAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/kernel/v
?
[Adam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/kernel/v* 
_output_shapes
:
?	?*
dtype0
?
EAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*V
shared_nameGEAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/bias/v
?
YAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/bias/v/Read/ReadVariableOpReadVariableOpEAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/bias/v*
_output_shapes	
:?*
dtype0
?
GAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*X
shared_nameIGAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/kernel/v
?
[Adam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/kernel/v* 
_output_shapes
:
??*
dtype0
?
EAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*V
shared_nameGEAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/bias/v
?
YAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/bias/v/Read/ReadVariableOpReadVariableOpEAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/bias/v*
_output_shapes	
:?*
dtype0
?
IAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*Z
shared_nameKIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/kernel/v
?
]Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/kernel/v*'
_output_shapes
:?*
dtype0
?
GAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*X
shared_nameIGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/bias/v
?
[Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/bias/v*
_output_shapes	
:?*
dtype0
?
IAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*Z
shared_nameKIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/kernel/v
?
]Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/kernel/v*(
_output_shapes
:??*
dtype0
?
GAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*X
shared_nameIGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/bias/v
?
[Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/bias/v*
_output_shapes	
:?*
dtype0
?
IAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*Z
shared_nameKIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/kernel/v
?
]Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/kernel/v*'
_output_shapes
:?@*
dtype0
?
GAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*X
shared_nameIGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/bias/v
?
[Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/bias/v*
_output_shapes
:@*
dtype0
?
IAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *Z
shared_nameKIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/kernel/v
?
]Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/kernel/v*&
_output_shapes
:@ *
dtype0
?
GAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/bias/v
?
[Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/bias/v*
_output_shapes
: *
dtype0
?
IAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Z
shared_nameKIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/kernel/v
?
]Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/kernel/v*&
_output_shapes
: *
dtype0
?
GAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*X
shared_nameIGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/bias/v
?
[Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*͕
valueB?? B??
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
VARIABLE_VALUEBconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_4/convolution_encoder_4/dense_8/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>convolutional_autoencoder_4/convolution_encoder_4/dense_8/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_4/convolution_decoder_4/dense_9/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>convolutional_autoencoder_4/convolution_decoder_4/dense_9/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/kernel1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/bias1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEEAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEEAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/kernel/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/bias/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/kernel/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/bias/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/kernel/mMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/bias/mMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/kernel/mMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/bias/mMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEEAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEEAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/kernel/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/bias/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/kernel/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/bias/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/kernel/vMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/bias/vMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/kernel/vMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/bias/vMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????@@*
dtype0*$
shape:?????????@@
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Bconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/kernel@convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/biasBconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/kernel@convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/biasBconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/kernel@convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/biasBconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/kernel@convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/bias@convolutional_autoencoder_4/convolution_encoder_4/dense_8/kernel>convolutional_autoencoder_4/convolution_encoder_4/dense_8/bias@convolutional_autoencoder_4/convolution_decoder_4/dense_9/kernel>convolutional_autoencoder_4/convolution_decoder_4/dense_9/biasBconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/kernel@convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/biasBconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/kernel@convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/biasBconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/kernel@convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/biasBconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/kernel@convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/biasBconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/kernel@convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/bias*"
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
#__inference_signature_wrapper_60559
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?3
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpVconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/kernel/Read/ReadVariableOpTconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/bias/Read/ReadVariableOpVconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/kernel/Read/ReadVariableOpTconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/bias/Read/ReadVariableOpVconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/kernel/Read/ReadVariableOpTconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/bias/Read/ReadVariableOpVconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/kernel/Read/ReadVariableOpTconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/bias/Read/ReadVariableOpTconvolutional_autoencoder_4/convolution_encoder_4/dense_8/kernel/Read/ReadVariableOpRconvolutional_autoencoder_4/convolution_encoder_4/dense_8/bias/Read/ReadVariableOpTconvolutional_autoencoder_4/convolution_decoder_4/dense_9/kernel/Read/ReadVariableOpRconvolutional_autoencoder_4/convolution_decoder_4/dense_9/bias/Read/ReadVariableOpVconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/kernel/Read/ReadVariableOpTconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/bias/Read/ReadVariableOpVconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/kernel/Read/ReadVariableOpTconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/bias/Read/ReadVariableOpVconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/kernel/Read/ReadVariableOpTconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/bias/Read/ReadVariableOpVconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/kernel/Read/ReadVariableOpTconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/bias/Read/ReadVariableOpVconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/kernel/Read/ReadVariableOpTconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp]Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/bias/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/kernel/m/Read/ReadVariableOpYAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/bias/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/kernel/m/Read/ReadVariableOpYAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/bias/v/Read/ReadVariableOp]Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/bias/v/Read/ReadVariableOp]Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/bias/v/Read/ReadVariableOp]Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/bias/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/kernel/v/Read/ReadVariableOpYAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/bias/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/kernel/v/Read/ReadVariableOpYAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/bias/v/Read/ReadVariableOp]Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/bias/v/Read/ReadVariableOp]Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/bias/v/Read/ReadVariableOp]Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/bias/v/Read/ReadVariableOp]Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/bias/v/Read/ReadVariableOp]Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/bias/v/Read/ReadVariableOpConst*V
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
__inference__traced_save_61048
?(
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateBconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/kernel@convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/biasBconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/kernel@convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/biasBconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/kernel@convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/biasBconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/kernel@convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/bias@convolutional_autoencoder_4/convolution_encoder_4/dense_8/kernel>convolutional_autoencoder_4/convolution_encoder_4/dense_8/bias@convolutional_autoencoder_4/convolution_decoder_4/dense_9/kernel>convolutional_autoencoder_4/convolution_decoder_4/dense_9/biasBconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/kernel@convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/biasBconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/kernel@convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/biasBconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/kernel@convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/biasBconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/kernel@convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/biasBconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/kernel@convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/biastotalcountIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/kernel/mGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/bias/mIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/kernel/mGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/bias/mIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/kernel/mGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/bias/mIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/kernel/mGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/bias/mGAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/kernel/mEAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/bias/mGAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/kernel/mEAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/bias/mIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/kernel/mGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/bias/mIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/kernel/mGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/bias/mIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/kernel/mGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/bias/mIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/kernel/mGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/bias/mIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/kernel/mGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/bias/mIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/kernel/vGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/bias/vIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/kernel/vGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/bias/vIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/kernel/vGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/bias/vIAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/kernel/vGAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/bias/vGAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/kernel/vEAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/bias/vGAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/kernel/vEAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/bias/vIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/kernel/vGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/bias/vIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/kernel/vGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/bias/vIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/kernel/vGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/bias/vIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/kernel/vGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/bias/vIAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/kernel/vGAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/bias/v*U
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
!__inference__traced_restore_61277??
?	
?
B__inference_dense_8_layer_call_and_return_conditional_losses_60580

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

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
?
~
)__inference_conv2d_40_layer_call_fn_60746

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
D__inference_conv2d_40_layer_call_and_return_conditional_losses_602412
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
?
|
'__inference_dense_8_layer_call_fn_60589

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
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_600592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
`
D__inference_flatten_4_layer_call_and_return_conditional_losses_60041

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
)__inference_conv2d_36_layer_call_fn_60666

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
GPU 2J 8? *M
fHRF
D__inference_conv2d_36_layer_call_and_return_conditional_losses_599382
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
?
?
V__inference_convolutional_autoencoder_4_layer_call_and_return_conditional_losses_60450
input_1
convolution_encoder_4_60403
convolution_encoder_4_60405
convolution_encoder_4_60407
convolution_encoder_4_60409
convolution_encoder_4_60411
convolution_encoder_4_60413
convolution_encoder_4_60415
convolution_encoder_4_60417
convolution_encoder_4_60419
convolution_encoder_4_60421
convolution_decoder_4_60424
convolution_decoder_4_60426
convolution_decoder_4_60428
convolution_decoder_4_60430
convolution_decoder_4_60432
convolution_decoder_4_60434
convolution_decoder_4_60436
convolution_decoder_4_60438
convolution_decoder_4_60440
convolution_decoder_4_60442
convolution_decoder_4_60444
convolution_decoder_4_60446
identity??-convolution_decoder_4/StatefulPartitionedCall?-convolution_encoder_4/StatefulPartitionedCall?
-convolution_encoder_4/StatefulPartitionedCallStatefulPartitionedCallinput_1convolution_encoder_4_60403convolution_encoder_4_60405convolution_encoder_4_60407convolution_encoder_4_60409convolution_encoder_4_60411convolution_encoder_4_60413convolution_encoder_4_60415convolution_encoder_4_60417convolution_encoder_4_60419convolution_encoder_4_60421*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*,
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
P__inference_convolution_encoder_4_layer_call_and_return_conditional_losses_600762/
-convolution_encoder_4/StatefulPartitionedCall?
-convolution_decoder_4/StatefulPartitionedCallStatefulPartitionedCall6convolution_encoder_4/StatefulPartitionedCall:output:0convolution_decoder_4_60424convolution_decoder_4_60426convolution_decoder_4_60428convolution_decoder_4_60430convolution_decoder_4_60432convolution_decoder_4_60434convolution_decoder_4_60436convolution_decoder_4_60438convolution_decoder_4_60440convolution_decoder_4_60442convolution_decoder_4_60444convolution_decoder_4_60446*
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
P__inference_convolution_decoder_4_layer_call_and_return_conditional_losses_603692/
-convolution_decoder_4/StatefulPartitionedCall?
IdentityIdentity6convolution_decoder_4/StatefulPartitionedCall:output:0.^convolution_decoder_4/StatefulPartitionedCall.^convolution_encoder_4/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????@@::::::::::::::::::::::2^
-convolution_decoder_4/StatefulPartitionedCall-convolution_decoder_4/StatefulPartitionedCall2^
-convolution_encoder_4/StatefulPartitionedCall-convolution_encoder_4/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?
~
)__inference_conv2d_39_layer_call_fn_60726

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
D__inference_conv2d_39_layer_call_and_return_conditional_losses_600192
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
?
~
)__inference_conv2d_44_layer_call_fn_60646

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
D__inference_conv2d_44_layer_call_and_return_conditional_losses_603522
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
g
K__inference_up_sampling2d_17_layer_call_and_return_conditional_losses_60134

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
~
)__inference_conv2d_37_layer_call_fn_60686

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
D__inference_conv2d_37_layer_call_and_return_conditional_losses_599652
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
g
K__inference_up_sampling2d_16_layer_call_and_return_conditional_losses_60115

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
D__inference_conv2d_36_layer_call_and_return_conditional_losses_60657

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
B__inference_dense_9_layer_call_and_return_conditional_losses_60599

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
5__inference_convolution_decoder_4_layer_call_fn_60399
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
P__inference_convolution_decoder_4_layer_call_and_return_conditional_losses_603692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:??????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
;__inference_convolutional_autoencoder_4_layer_call_fn_60500
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
V__inference_convolutional_autoencoder_4_layer_call_and_return_conditional_losses_604502
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
?
E
)__inference_reshape_4_layer_call_fn_60627

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
D__inference_reshape_4_layer_call_and_return_conditional_losses_602222
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
g
K__inference_up_sampling2d_19_layer_call_and_return_conditional_losses_60172

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
D__inference_conv2d_36_layer_call_and_return_conditional_losses_59938

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
D__inference_conv2d_38_layer_call_and_return_conditional_losses_60697

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
)__inference_conv2d_41_layer_call_fn_60766

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
D__inference_conv2d_41_layer_call_and_return_conditional_losses_602692
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
?
|
'__inference_dense_9_layer_call_fn_60608

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
B__inference_dense_9_layer_call_and_return_conditional_losses_601922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
P__inference_convolution_encoder_4_layer_call_and_return_conditional_losses_60076
input_1
conv2d_36_59949
conv2d_36_59951
conv2d_37_59976
conv2d_37_59978
conv2d_38_60003
conv2d_38_60005
conv2d_39_60030
conv2d_39_60032
dense_8_60070
dense_8_60072
identity??!conv2d_36/StatefulPartitionedCall?!conv2d_37/StatefulPartitionedCall?!conv2d_38/StatefulPartitionedCall?!conv2d_39/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_36_59949conv2d_36_59951*
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
GPU 2J 8? *M
fHRF
D__inference_conv2d_36_layer_call_and_return_conditional_losses_599382#
!conv2d_36/StatefulPartitionedCall?
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0conv2d_37_59976conv2d_37_59978*
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
D__inference_conv2d_37_layer_call_and_return_conditional_losses_599652#
!conv2d_37/StatefulPartitionedCall?
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0conv2d_38_60003conv2d_38_60005*
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
D__inference_conv2d_38_layer_call_and_return_conditional_losses_599922#
!conv2d_38/StatefulPartitionedCall?
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0conv2d_39_60030conv2d_39_60032*
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
D__inference_conv2d_39_layer_call_and_return_conditional_losses_600192#
!conv2d_39/StatefulPartitionedCall?
flatten_4/PartitionedCallPartitionedCall*conv2d_39/StatefulPartitionedCall:output:0*
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
D__inference_flatten_4_layer_call_and_return_conditional_losses_600412
flatten_4/PartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_8_60070dense_8_60072*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_600592!
dense_8/StatefulPartitionedCall?
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall"^conv2d_39/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????@@::::::::::2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?
?
D__inference_conv2d_41_layer_call_and_return_conditional_losses_60269

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

?
D__inference_conv2d_44_layer_call_and_return_conditional_losses_60637

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
g
K__inference_up_sampling2d_18_layer_call_and_return_conditional_losses_60153

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
??
?
 __inference__wrapped_model_59923
input_1^
Zconvolutional_autoencoder_4_convolution_encoder_4_conv2d_36_conv2d_readvariableop_resource_
[convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_biasadd_readvariableop_resource^
Zconvolutional_autoencoder_4_convolution_encoder_4_conv2d_37_conv2d_readvariableop_resource_
[convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_biasadd_readvariableop_resource^
Zconvolutional_autoencoder_4_convolution_encoder_4_conv2d_38_conv2d_readvariableop_resource_
[convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_biasadd_readvariableop_resource^
Zconvolutional_autoencoder_4_convolution_encoder_4_conv2d_39_conv2d_readvariableop_resource_
[convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_biasadd_readvariableop_resource\
Xconvolutional_autoencoder_4_convolution_encoder_4_dense_8_matmul_readvariableop_resource]
Yconvolutional_autoencoder_4_convolution_encoder_4_dense_8_biasadd_readvariableop_resource\
Xconvolutional_autoencoder_4_convolution_decoder_4_dense_9_matmul_readvariableop_resource]
Yconvolutional_autoencoder_4_convolution_decoder_4_dense_9_biasadd_readvariableop_resource^
Zconvolutional_autoencoder_4_convolution_decoder_4_conv2d_40_conv2d_readvariableop_resource_
[convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_biasadd_readvariableop_resource^
Zconvolutional_autoencoder_4_convolution_decoder_4_conv2d_41_conv2d_readvariableop_resource_
[convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_biasadd_readvariableop_resource^
Zconvolutional_autoencoder_4_convolution_decoder_4_conv2d_42_conv2d_readvariableop_resource_
[convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_biasadd_readvariableop_resource^
Zconvolutional_autoencoder_4_convolution_decoder_4_conv2d_43_conv2d_readvariableop_resource_
[convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_biasadd_readvariableop_resource^
Zconvolutional_autoencoder_4_convolution_decoder_4_conv2d_44_conv2d_readvariableop_resource_
[convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_biasadd_readvariableop_resource
identity??Rconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/Conv2D/ReadVariableOp?Rconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/Conv2D/ReadVariableOp?Rconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/Conv2D/ReadVariableOp?Rconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/Conv2D/ReadVariableOp?Rconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/Conv2D/ReadVariableOp?Pconvolutional_autoencoder_4/convolution_decoder_4/dense_9/BiasAdd/ReadVariableOp?Oconvolutional_autoencoder_4/convolution_decoder_4/dense_9/MatMul/ReadVariableOp?Rconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/Conv2D/ReadVariableOp?Rconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/Conv2D/ReadVariableOp?Rconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/Conv2D/ReadVariableOp?Rconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/Conv2D/ReadVariableOp?Pconvolutional_autoencoder_4/convolution_encoder_4/dense_8/BiasAdd/ReadVariableOp?Oconvolutional_autoencoder_4/convolution_encoder_4/dense_8/MatMul/ReadVariableOp?
Qconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_4_convolution_encoder_4_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02S
Qconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/Conv2DConv2Dinput_1Yconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2D
Bconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/Conv2D?
Rconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02T
Rconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/BiasAddBiasAddKconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/Conv2D:output:0Zconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2E
Cconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/BiasAdd?
@convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/ReluReluLconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2B
@convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/Relu?
Qconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_4_convolution_encoder_4_conv2d_37_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02S
Qconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/Conv2DConv2DNconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/Relu:activations:0Yconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2D
Bconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/Conv2D?
Rconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02T
Rconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/BiasAddBiasAddKconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/Conv2D:output:0Zconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2E
Cconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/BiasAdd?
@convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/ReluReluLconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2B
@convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/Relu?
Qconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_4_convolution_encoder_4_conv2d_38_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02S
Qconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/Conv2DConv2DNconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/Relu:activations:0Yconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2D
Bconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/Conv2D?
Rconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02T
Rconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/BiasAddBiasAddKconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/Conv2D:output:0Zconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2E
Cconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/BiasAdd?
@convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/ReluReluLconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2B
@convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/Relu?
Qconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_4_convolution_encoder_4_conv2d_39_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02S
Qconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/Conv2DConv2DNconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/Relu:activations:0Yconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2D
Bconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/Conv2D?
Rconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02T
Rconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/BiasAddBiasAddKconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/Conv2D:output:0Zconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2E
Cconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/BiasAdd?
@convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/ReluReluLconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2B
@convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/Relu?
Aconvolutional_autoencoder_4/convolution_encoder_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2C
Aconvolutional_autoencoder_4/convolution_encoder_4/flatten_4/Const?
Cconvolutional_autoencoder_4/convolution_encoder_4/flatten_4/ReshapeReshapeNconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/Relu:activations:0Jconvolutional_autoencoder_4/convolution_encoder_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????	2E
Cconvolutional_autoencoder_4/convolution_encoder_4/flatten_4/Reshape?
Oconvolutional_autoencoder_4/convolution_encoder_4/dense_8/MatMul/ReadVariableOpReadVariableOpXconvolutional_autoencoder_4_convolution_encoder_4_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02Q
Oconvolutional_autoencoder_4/convolution_encoder_4/dense_8/MatMul/ReadVariableOp?
@convolutional_autoencoder_4/convolution_encoder_4/dense_8/MatMulMatMulLconvolutional_autoencoder_4/convolution_encoder_4/flatten_4/Reshape:output:0Wconvolutional_autoencoder_4/convolution_encoder_4/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2B
@convolutional_autoencoder_4/convolution_encoder_4/dense_8/MatMul?
Pconvolutional_autoencoder_4/convolution_encoder_4/dense_8/BiasAdd/ReadVariableOpReadVariableOpYconvolutional_autoencoder_4_convolution_encoder_4_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02R
Pconvolutional_autoencoder_4/convolution_encoder_4/dense_8/BiasAdd/ReadVariableOp?
Aconvolutional_autoencoder_4/convolution_encoder_4/dense_8/BiasAddBiasAddJconvolutional_autoencoder_4/convolution_encoder_4/dense_8/MatMul:product:0Xconvolutional_autoencoder_4/convolution_encoder_4/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2C
Aconvolutional_autoencoder_4/convolution_encoder_4/dense_8/BiasAdd?
Oconvolutional_autoencoder_4/convolution_decoder_4/dense_9/MatMul/ReadVariableOpReadVariableOpXconvolutional_autoencoder_4_convolution_decoder_4_dense_9_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02Q
Oconvolutional_autoencoder_4/convolution_decoder_4/dense_9/MatMul/ReadVariableOp?
@convolutional_autoencoder_4/convolution_decoder_4/dense_9/MatMulMatMulJconvolutional_autoencoder_4/convolution_encoder_4/dense_8/BiasAdd:output:0Wconvolutional_autoencoder_4/convolution_decoder_4/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2B
@convolutional_autoencoder_4/convolution_decoder_4/dense_9/MatMul?
Pconvolutional_autoencoder_4/convolution_decoder_4/dense_9/BiasAdd/ReadVariableOpReadVariableOpYconvolutional_autoencoder_4_convolution_decoder_4_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02R
Pconvolutional_autoencoder_4/convolution_decoder_4/dense_9/BiasAdd/ReadVariableOp?
Aconvolutional_autoencoder_4/convolution_decoder_4/dense_9/BiasAddBiasAddJconvolutional_autoencoder_4/convolution_decoder_4/dense_9/MatMul:product:0Xconvolutional_autoencoder_4/convolution_decoder_4/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2C
Aconvolutional_autoencoder_4/convolution_decoder_4/dense_9/BiasAdd?
Aconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/ShapeShapeJconvolutional_autoencoder_4/convolution_decoder_4/dense_9/BiasAdd:output:0*
T0*
_output_shapes
:2C
Aconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/Shape?
Oconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Q
Oconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/strided_slice/stack?
Qconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2S
Qconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/strided_slice/stack_1?
Qconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
Qconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/strided_slice/stack_2?
Iconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/strided_sliceStridedSliceJconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/Shape:output:0Xconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/strided_slice/stack:output:0Zconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/strided_slice/stack_1:output:0Zconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2K
Iconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/strided_slice?
Kconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2M
Kconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/Reshape/shape/1?
Kconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2M
Kconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/Reshape/shape/2?
Kconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2M
Kconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/Reshape/shape/3?
Iconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/Reshape/shapePackRconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/strided_slice:output:0Tconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/Reshape/shape/1:output:0Tconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/Reshape/shape/2:output:0Tconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2K
Iconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/Reshape/shape?
Cconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/ReshapeReshapeJconvolutional_autoencoder_4/convolution_decoder_4/dense_9/BiasAdd:output:0Rconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2E
Cconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/Reshape?
Qconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_4_convolution_decoder_4_conv2d_40_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02S
Qconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/Conv2DConv2DLconvolutional_autoencoder_4/convolution_decoder_4/reshape_4/Reshape:output:0Yconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2D
Bconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/Conv2D?
Rconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02T
Rconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/BiasAddBiasAddKconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/Conv2D:output:0Zconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2E
Cconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/BiasAdd?
@convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/ReluReluLconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2B
@convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/Relu?
Hconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_16/ShapeShapeNconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/Relu:activations:0*
T0*
_output_shapes
:2J
Hconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_16/Shape?
Vconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2X
Vconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_16/strided_slice/stack?
Xconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Z
Xconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_16/strided_slice/stack_1?
Xconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Z
Xconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_16/strided_slice/stack_2?
Pconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_16/strided_sliceStridedSliceQconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_16/Shape:output:0_convolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_16/strided_slice/stack:output:0aconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_16/strided_slice/stack_1:output:0aconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2R
Pconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_16/strided_slice?
Hconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2J
Hconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_16/Const?
Fconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_16/mulMulYconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_16/strided_slice:output:0Qconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_16/Const:output:0*
T0*
_output_shapes
:2H
Fconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_16/mul?
_convolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_16/resize/ResizeNearestNeighborResizeNearestNeighborNconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/Relu:activations:0Jconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_16/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2a
_convolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_16/resize/ResizeNearestNeighbor?
Qconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_4_convolution_decoder_4_conv2d_41_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02S
Qconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/Conv2DConv2Dpconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_16/resize/ResizeNearestNeighbor:resized_images:0Yconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2D
Bconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/Conv2D?
Rconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02T
Rconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/BiasAddBiasAddKconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/Conv2D:output:0Zconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2E
Cconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/BiasAdd?
@convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/ReluReluLconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2B
@convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/Relu?
Hconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_17/ShapeShapeNconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/Relu:activations:0*
T0*
_output_shapes
:2J
Hconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_17/Shape?
Vconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2X
Vconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_17/strided_slice/stack?
Xconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Z
Xconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_17/strided_slice/stack_1?
Xconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Z
Xconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_17/strided_slice/stack_2?
Pconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_17/strided_sliceStridedSliceQconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_17/Shape:output:0_convolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_17/strided_slice/stack:output:0aconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_17/strided_slice/stack_1:output:0aconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2R
Pconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_17/strided_slice?
Hconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2J
Hconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_17/Const?
Fconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_17/mulMulYconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_17/strided_slice:output:0Qconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_17/Const:output:0*
T0*
_output_shapes
:2H
Fconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_17/mul?
_convolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_17/resize/ResizeNearestNeighborResizeNearestNeighborNconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/Relu:activations:0Jconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_17/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2a
_convolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_17/resize/ResizeNearestNeighbor?
Qconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_4_convolution_decoder_4_conv2d_42_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02S
Qconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/Conv2DConv2Dpconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_17/resize/ResizeNearestNeighbor:resized_images:0Yconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2D
Bconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/Conv2D?
Rconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02T
Rconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/BiasAddBiasAddKconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/Conv2D:output:0Zconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2E
Cconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/BiasAdd?
@convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/ReluReluLconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2B
@convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/Relu?
Hconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_18/ShapeShapeNconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/Relu:activations:0*
T0*
_output_shapes
:2J
Hconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_18/Shape?
Vconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2X
Vconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_18/strided_slice/stack?
Xconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Z
Xconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_18/strided_slice/stack_1?
Xconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Z
Xconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_18/strided_slice/stack_2?
Pconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_18/strided_sliceStridedSliceQconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_18/Shape:output:0_convolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_18/strided_slice/stack:output:0aconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_18/strided_slice/stack_1:output:0aconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2R
Pconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_18/strided_slice?
Hconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2J
Hconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_18/Const?
Fconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_18/mulMulYconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_18/strided_slice:output:0Qconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_18/Const:output:0*
T0*
_output_shapes
:2H
Fconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_18/mul?
_convolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_18/resize/ResizeNearestNeighborResizeNearestNeighborNconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/Relu:activations:0Jconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_18/mul:z:0*
T0*/
_output_shapes
:?????????  @*
half_pixel_centers(2a
_convolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_18/resize/ResizeNearestNeighbor?
Qconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_4_convolution_decoder_4_conv2d_43_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02S
Qconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/Conv2DConv2Dpconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_18/resize/ResizeNearestNeighbor:resized_images:0Yconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2D
Bconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/Conv2D?
Rconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02T
Rconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/BiasAddBiasAddKconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/Conv2D:output:0Zconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2E
Cconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/BiasAdd?
@convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/ReluReluLconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2B
@convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/Relu?
Hconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_19/ShapeShapeNconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/Relu:activations:0*
T0*
_output_shapes
:2J
Hconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_19/Shape?
Vconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2X
Vconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_19/strided_slice/stack?
Xconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Z
Xconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_19/strided_slice/stack_1?
Xconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Z
Xconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_19/strided_slice/stack_2?
Pconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_19/strided_sliceStridedSliceQconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_19/Shape:output:0_convolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_19/strided_slice/stack:output:0aconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_19/strided_slice/stack_1:output:0aconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2R
Pconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_19/strided_slice?
Hconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_19/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2J
Hconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_19/Const?
Fconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_19/mulMulYconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_19/strided_slice:output:0Qconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_19/Const:output:0*
T0*
_output_shapes
:2H
Fconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_19/mul?
_convolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_19/resize/ResizeNearestNeighborResizeNearestNeighborNconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/Relu:activations:0Jconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_19/mul:z:0*
T0*/
_output_shapes
:?????????@@ *
half_pixel_centers(2a
_convolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_19/resize/ResizeNearestNeighbor?
Qconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_4_convolution_decoder_4_conv2d_44_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02S
Qconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/Conv2DConv2Dpconvolutional_autoencoder_4/convolution_decoder_4/up_sampling2d_19/resize/ResizeNearestNeighbor:resized_images:0Yconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2D
Bconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/Conv2D?
Rconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02T
Rconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/BiasAddBiasAddKconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/Conv2D:output:0Zconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2E
Cconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/BiasAdd?
IdentityIdentityLconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/BiasAdd:output:0S^convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/BiasAdd/ReadVariableOpR^convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/Conv2D/ReadVariableOpS^convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/BiasAdd/ReadVariableOpR^convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/Conv2D/ReadVariableOpS^convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/BiasAdd/ReadVariableOpR^convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/Conv2D/ReadVariableOpS^convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/BiasAdd/ReadVariableOpR^convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/Conv2D/ReadVariableOpS^convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/BiasAdd/ReadVariableOpR^convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/Conv2D/ReadVariableOpQ^convolutional_autoencoder_4/convolution_decoder_4/dense_9/BiasAdd/ReadVariableOpP^convolutional_autoencoder_4/convolution_decoder_4/dense_9/MatMul/ReadVariableOpS^convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/BiasAdd/ReadVariableOpR^convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/Conv2D/ReadVariableOpS^convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/BiasAdd/ReadVariableOpR^convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/Conv2D/ReadVariableOpS^convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/BiasAdd/ReadVariableOpR^convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/Conv2D/ReadVariableOpS^convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/BiasAdd/ReadVariableOpR^convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/Conv2D/ReadVariableOpQ^convolutional_autoencoder_4/convolution_encoder_4/dense_8/BiasAdd/ReadVariableOpP^convolutional_autoencoder_4/convolution_encoder_4/dense_8/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????@@::::::::::::::::::::::2?
Rconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/BiasAdd/ReadVariableOpRconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/Conv2D/ReadVariableOpQconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/Conv2D/ReadVariableOp2?
Rconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/BiasAdd/ReadVariableOpRconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/Conv2D/ReadVariableOpQconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/Conv2D/ReadVariableOp2?
Rconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/BiasAdd/ReadVariableOpRconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/Conv2D/ReadVariableOpQconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/Conv2D/ReadVariableOp2?
Rconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/BiasAdd/ReadVariableOpRconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/Conv2D/ReadVariableOpQconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/Conv2D/ReadVariableOp2?
Rconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/BiasAdd/ReadVariableOpRconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/Conv2D/ReadVariableOpQconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/Conv2D/ReadVariableOp2?
Pconvolutional_autoencoder_4/convolution_decoder_4/dense_9/BiasAdd/ReadVariableOpPconvolutional_autoencoder_4/convolution_decoder_4/dense_9/BiasAdd/ReadVariableOp2?
Oconvolutional_autoencoder_4/convolution_decoder_4/dense_9/MatMul/ReadVariableOpOconvolutional_autoencoder_4/convolution_decoder_4/dense_9/MatMul/ReadVariableOp2?
Rconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/BiasAdd/ReadVariableOpRconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/Conv2D/ReadVariableOpQconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/Conv2D/ReadVariableOp2?
Rconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/BiasAdd/ReadVariableOpRconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/Conv2D/ReadVariableOpQconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/Conv2D/ReadVariableOp2?
Rconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/BiasAdd/ReadVariableOpRconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/Conv2D/ReadVariableOpQconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/Conv2D/ReadVariableOp2?
Rconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/BiasAdd/ReadVariableOpRconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/Conv2D/ReadVariableOpQconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/Conv2D/ReadVariableOp2?
Pconvolutional_autoencoder_4/convolution_encoder_4/dense_8/BiasAdd/ReadVariableOpPconvolutional_autoencoder_4/convolution_encoder_4/dense_8/BiasAdd/ReadVariableOp2?
Oconvolutional_autoencoder_4/convolution_encoder_4/dense_8/MatMul/ReadVariableOpOconvolutional_autoencoder_4/convolution_encoder_4/dense_8/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?

?
D__inference_conv2d_40_layer_call_and_return_conditional_losses_60737

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
D__inference_conv2d_39_layer_call_and_return_conditional_losses_60019

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
`
D__inference_reshape_4_layer_call_and_return_conditional_losses_60622

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
D__inference_conv2d_39_layer_call_and_return_conditional_losses_60717

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
B__inference_dense_9_layer_call_and_return_conditional_losses_60192

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_37_layer_call_and_return_conditional_losses_59965

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
D__inference_conv2d_43_layer_call_and_return_conditional_losses_60797

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
?
5__inference_convolution_encoder_4_layer_call_fn_60102
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
 *(
_output_shapes
:??????????*,
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
P__inference_convolution_encoder_4_layer_call_and_return_conditional_losses_600762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

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
D__inference_conv2d_40_layer_call_and_return_conditional_losses_60241

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
?
`
D__inference_flatten_4_layer_call_and_return_conditional_losses_60565

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
L
0__inference_up_sampling2d_17_layer_call_fn_60140

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
GPU 2J 8? *T
fORM
K__inference_up_sampling2d_17_layer_call_and_return_conditional_losses_601342
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
D__inference_conv2d_41_layer_call_and_return_conditional_losses_60757

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
?
B__inference_dense_8_layer_call_and_return_conditional_losses_60059

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

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
D__inference_conv2d_42_layer_call_and_return_conditional_losses_60777

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
?
~
)__inference_conv2d_38_layer_call_fn_60706

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
D__inference_conv2d_38_layer_call_and_return_conditional_losses_599922
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
#__inference_signature_wrapper_60559
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
 __inference__wrapped_model_599232
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
?

?
D__inference_conv2d_38_layer_call_and_return_conditional_losses_59992

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
L
0__inference_up_sampling2d_19_layer_call_fn_60178

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
GPU 2J 8? *T
fORM
K__inference_up_sampling2d_19_layer_call_and_return_conditional_losses_601722
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
)__inference_flatten_4_layer_call_fn_60570

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
D__inference_flatten_4_layer_call_and_return_conditional_losses_600412
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
?
L
0__inference_up_sampling2d_18_layer_call_fn_60159

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
GPU 2J 8? *T
fORM
K__inference_up_sampling2d_18_layer_call_and_return_conditional_losses_601532
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
?
~
)__inference_conv2d_42_layer_call_fn_60786

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
D__inference_conv2d_42_layer_call_and_return_conditional_losses_602972
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
?2
?
P__inference_convolution_decoder_4_layer_call_and_return_conditional_losses_60369
input_1
dense_9_60203
dense_9_60205
conv2d_40_60252
conv2d_40_60254
conv2d_41_60280
conv2d_41_60282
conv2d_42_60308
conv2d_42_60310
conv2d_43_60336
conv2d_43_60338
conv2d_44_60363
conv2d_44_60365
identity??!conv2d_40/StatefulPartitionedCall?!conv2d_41/StatefulPartitionedCall?!conv2d_42/StatefulPartitionedCall?!conv2d_43/StatefulPartitionedCall?!conv2d_44/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_9_60203dense_9_60205*
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
B__inference_dense_9_layer_call_and_return_conditional_losses_601922!
dense_9/StatefulPartitionedCall?
reshape_4/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
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
D__inference_reshape_4_layer_call_and_return_conditional_losses_602222
reshape_4/PartitionedCall?
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCall"reshape_4/PartitionedCall:output:0conv2d_40_60252conv2d_40_60254*
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
D__inference_conv2d_40_layer_call_and_return_conditional_losses_602412#
!conv2d_40/StatefulPartitionedCall?
 up_sampling2d_16/PartitionedCallPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_up_sampling2d_16_layer_call_and_return_conditional_losses_601152"
 up_sampling2d_16/PartitionedCall?
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_16/PartitionedCall:output:0conv2d_41_60280conv2d_41_60282*
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
D__inference_conv2d_41_layer_call_and_return_conditional_losses_602692#
!conv2d_41/StatefulPartitionedCall?
 up_sampling2d_17/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_up_sampling2d_17_layer_call_and_return_conditional_losses_601342"
 up_sampling2d_17/PartitionedCall?
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_17/PartitionedCall:output:0conv2d_42_60308conv2d_42_60310*
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
D__inference_conv2d_42_layer_call_and_return_conditional_losses_602972#
!conv2d_42/StatefulPartitionedCall?
 up_sampling2d_18/PartitionedCallPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_up_sampling2d_18_layer_call_and_return_conditional_losses_601532"
 up_sampling2d_18/PartitionedCall?
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_18/PartitionedCall:output:0conv2d_43_60336conv2d_43_60338*
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
D__inference_conv2d_43_layer_call_and_return_conditional_losses_603252#
!conv2d_43/StatefulPartitionedCall?
 up_sampling2d_19/PartitionedCallPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_up_sampling2d_19_layer_call_and_return_conditional_losses_601722"
 up_sampling2d_19/PartitionedCall?
!conv2d_44/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_19/PartitionedCall:output:0conv2d_44_60363conv2d_44_60365*
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
D__inference_conv2d_44_layer_call_and_return_conditional_losses_603522#
!conv2d_44/StatefulPartitionedCall?
IdentityIdentity*conv2d_44/StatefulPartitionedCall:output:0"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall"^conv2d_44/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:??????????::::::::::::2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2F
!conv2d_44/StatefulPartitionedCall!conv2d_44/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
`
D__inference_reshape_4_layer_call_and_return_conditional_losses_60222

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
?
~
)__inference_conv2d_43_layer_call_fn_60806

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
D__inference_conv2d_43_layer_call_and_return_conditional_losses_603252
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
??
?@
!__inference__traced_restore_61277
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rateY
Uassignvariableop_5_convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_kernelW
Sassignvariableop_6_convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_biasY
Uassignvariableop_7_convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_kernelW
Sassignvariableop_8_convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_biasY
Uassignvariableop_9_convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_kernelX
Tassignvariableop_10_convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_biasZ
Vassignvariableop_11_convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_kernelX
Tassignvariableop_12_convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_biasX
Tassignvariableop_13_convolutional_autoencoder_4_convolution_encoder_4_dense_8_kernelV
Rassignvariableop_14_convolutional_autoencoder_4_convolution_encoder_4_dense_8_biasX
Tassignvariableop_15_convolutional_autoencoder_4_convolution_decoder_4_dense_9_kernelV
Rassignvariableop_16_convolutional_autoencoder_4_convolution_decoder_4_dense_9_biasZ
Vassignvariableop_17_convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_kernelX
Tassignvariableop_18_convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_biasZ
Vassignvariableop_19_convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_kernelX
Tassignvariableop_20_convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_biasZ
Vassignvariableop_21_convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_kernelX
Tassignvariableop_22_convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_biasZ
Vassignvariableop_23_convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_kernelX
Tassignvariableop_24_convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_biasZ
Vassignvariableop_25_convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_kernelX
Tassignvariableop_26_convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_bias
assignvariableop_27_total
assignvariableop_28_counta
]assignvariableop_29_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_kernel_m_
[assignvariableop_30_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_bias_ma
]assignvariableop_31_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_kernel_m_
[assignvariableop_32_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_bias_ma
]assignvariableop_33_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_kernel_m_
[assignvariableop_34_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_bias_ma
]assignvariableop_35_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_kernel_m_
[assignvariableop_36_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_bias_m_
[assignvariableop_37_adam_convolutional_autoencoder_4_convolution_encoder_4_dense_8_kernel_m]
Yassignvariableop_38_adam_convolutional_autoencoder_4_convolution_encoder_4_dense_8_bias_m_
[assignvariableop_39_adam_convolutional_autoencoder_4_convolution_decoder_4_dense_9_kernel_m]
Yassignvariableop_40_adam_convolutional_autoencoder_4_convolution_decoder_4_dense_9_bias_ma
]assignvariableop_41_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_kernel_m_
[assignvariableop_42_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_bias_ma
]assignvariableop_43_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_kernel_m_
[assignvariableop_44_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_bias_ma
]assignvariableop_45_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_kernel_m_
[assignvariableop_46_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_bias_ma
]assignvariableop_47_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_kernel_m_
[assignvariableop_48_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_bias_ma
]assignvariableop_49_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_kernel_m_
[assignvariableop_50_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_bias_ma
]assignvariableop_51_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_kernel_v_
[assignvariableop_52_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_bias_va
]assignvariableop_53_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_kernel_v_
[assignvariableop_54_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_bias_va
]assignvariableop_55_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_kernel_v_
[assignvariableop_56_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_bias_va
]assignvariableop_57_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_kernel_v_
[assignvariableop_58_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_bias_v_
[assignvariableop_59_adam_convolutional_autoencoder_4_convolution_encoder_4_dense_8_kernel_v]
Yassignvariableop_60_adam_convolutional_autoencoder_4_convolution_encoder_4_dense_8_bias_v_
[assignvariableop_61_adam_convolutional_autoencoder_4_convolution_decoder_4_dense_9_kernel_v]
Yassignvariableop_62_adam_convolutional_autoencoder_4_convolution_decoder_4_dense_9_bias_va
]assignvariableop_63_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_kernel_v_
[assignvariableop_64_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_bias_va
]assignvariableop_65_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_kernel_v_
[assignvariableop_66_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_bias_va
]assignvariableop_67_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_kernel_v_
[assignvariableop_68_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_bias_va
]assignvariableop_69_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_kernel_v_
[assignvariableop_70_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_bias_va
]assignvariableop_71_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_kernel_v_
[assignvariableop_72_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_bias_v
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
AssignVariableOp_5AssignVariableOpUassignvariableop_5_convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpSassignvariableop_6_convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpUassignvariableop_7_convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpSassignvariableop_8_convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpUassignvariableop_9_convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpTassignvariableop_10_convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpVassignvariableop_11_convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpTassignvariableop_12_convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpTassignvariableop_13_convolutional_autoencoder_4_convolution_encoder_4_dense_8_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpRassignvariableop_14_convolutional_autoencoder_4_convolution_encoder_4_dense_8_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpTassignvariableop_15_convolutional_autoencoder_4_convolution_decoder_4_dense_9_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpRassignvariableop_16_convolutional_autoencoder_4_convolution_decoder_4_dense_9_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpVassignvariableop_17_convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpTassignvariableop_18_convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpVassignvariableop_19_convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpTassignvariableop_20_convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpVassignvariableop_21_convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpTassignvariableop_22_convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpVassignvariableop_23_convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpTassignvariableop_24_convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpVassignvariableop_25_convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpTassignvariableop_26_convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_biasIdentity_26:output:0"/device:CPU:0*
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
AssignVariableOp_29AssignVariableOp]assignvariableop_29_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp[assignvariableop_30_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp]assignvariableop_31_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp[assignvariableop_32_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp]assignvariableop_33_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp[assignvariableop_34_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp]assignvariableop_35_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp[assignvariableop_36_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp[assignvariableop_37_adam_convolutional_autoencoder_4_convolution_encoder_4_dense_8_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpYassignvariableop_38_adam_convolutional_autoencoder_4_convolution_encoder_4_dense_8_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp[assignvariableop_39_adam_convolutional_autoencoder_4_convolution_decoder_4_dense_9_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpYassignvariableop_40_adam_convolutional_autoencoder_4_convolution_decoder_4_dense_9_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp]assignvariableop_41_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp[assignvariableop_42_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp]assignvariableop_43_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp[assignvariableop_44_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp]assignvariableop_45_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp[assignvariableop_46_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp]assignvariableop_47_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp[assignvariableop_48_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp]assignvariableop_49_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp[assignvariableop_50_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp]assignvariableop_51_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp[assignvariableop_52_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp]assignvariableop_53_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp[assignvariableop_54_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp]assignvariableop_55_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp[assignvariableop_56_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp]assignvariableop_57_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp[assignvariableop_58_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp[assignvariableop_59_adam_convolutional_autoencoder_4_convolution_encoder_4_dense_8_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOpYassignvariableop_60_adam_convolutional_autoencoder_4_convolution_encoder_4_dense_8_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp[assignvariableop_61_adam_convolutional_autoencoder_4_convolution_decoder_4_dense_9_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOpYassignvariableop_62_adam_convolutional_autoencoder_4_convolution_decoder_4_dense_9_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp]assignvariableop_63_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp[assignvariableop_64_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp]assignvariableop_65_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp[assignvariableop_66_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp]assignvariableop_67_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp[assignvariableop_68_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp]assignvariableop_69_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp[assignvariableop_70_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp]assignvariableop_71_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp[assignvariableop_72_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_bias_vIdentity_72:output:0"/device:CPU:0*
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
?
D__inference_conv2d_43_layer_call_and_return_conditional_losses_60325

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
D__inference_conv2d_44_layer_call_and_return_conditional_losses_60352

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
D__inference_conv2d_37_layer_call_and_return_conditional_losses_60677

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
L
0__inference_up_sampling2d_16_layer_call_fn_60121

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
GPU 2J 8? *T
fORM
K__inference_up_sampling2d_16_layer_call_and_return_conditional_losses_601152
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
??
?7
__inference__traced_save_61048
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopa
]savev2_convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_bias_read_readvariableopa
]savev2_convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_bias_read_readvariableopa
]savev2_convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_bias_read_readvariableopa
]savev2_convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_bias_read_readvariableop_
[savev2_convolutional_autoencoder_4_convolution_encoder_4_dense_8_kernel_read_readvariableop]
Ysavev2_convolutional_autoencoder_4_convolution_encoder_4_dense_8_bias_read_readvariableop_
[savev2_convolutional_autoencoder_4_convolution_decoder_4_dense_9_kernel_read_readvariableop]
Ysavev2_convolutional_autoencoder_4_convolution_decoder_4_dense_9_bias_read_readvariableopa
]savev2_convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_bias_read_readvariableopa
]savev2_convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_bias_read_readvariableopa
]savev2_convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_bias_read_readvariableopa
]savev2_convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_bias_read_readvariableopa
]savev2_convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_bias_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_dense_8_kernel_m_read_readvariableopd
`savev2_adam_convolutional_autoencoder_4_convolution_encoder_4_dense_8_bias_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_dense_9_kernel_m_read_readvariableopd
`savev2_adam_convolutional_autoencoder_4_convolution_decoder_4_dense_9_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_bias_v_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_bias_v_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_bias_v_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_bias_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_dense_8_kernel_v_read_readvariableopd
`savev2_adam_convolutional_autoencoder_4_convolution_encoder_4_dense_8_bias_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_dense_9_kernel_v_read_readvariableopd
`savev2_adam_convolutional_autoencoder_4_convolution_decoder_4_dense_9_bias_v_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_bias_v_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_bias_v_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_bias_v_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_bias_v_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop]savev2_convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_kernel_read_readvariableop[savev2_convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_bias_read_readvariableop]savev2_convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_kernel_read_readvariableop[savev2_convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_bias_read_readvariableop]savev2_convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_kernel_read_readvariableop[savev2_convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_bias_read_readvariableop]savev2_convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_kernel_read_readvariableop[savev2_convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_bias_read_readvariableop[savev2_convolutional_autoencoder_4_convolution_encoder_4_dense_8_kernel_read_readvariableopYsavev2_convolutional_autoencoder_4_convolution_encoder_4_dense_8_bias_read_readvariableop[savev2_convolutional_autoencoder_4_convolution_decoder_4_dense_9_kernel_read_readvariableopYsavev2_convolutional_autoencoder_4_convolution_decoder_4_dense_9_bias_read_readvariableop]savev2_convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_kernel_read_readvariableop[savev2_convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_bias_read_readvariableop]savev2_convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_kernel_read_readvariableop[savev2_convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_bias_read_readvariableop]savev2_convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_kernel_read_readvariableop[savev2_convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_bias_read_readvariableop]savev2_convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_kernel_read_readvariableop[savev2_convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_bias_read_readvariableop]savev2_convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_kernel_read_readvariableop[savev2_convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopdsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_bias_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_dense_8_kernel_m_read_readvariableop`savev2_adam_convolutional_autoencoder_4_convolution_encoder_4_dense_8_bias_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_dense_9_kernel_m_read_readvariableop`savev2_adam_convolutional_autoencoder_4_convolution_decoder_4_dense_9_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_36_bias_v_read_readvariableopdsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_37_bias_v_read_readvariableopdsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_38_bias_v_read_readvariableopdsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_conv2d_39_bias_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_4_convolution_encoder_4_dense_8_kernel_v_read_readvariableop`savev2_adam_convolutional_autoencoder_4_convolution_encoder_4_dense_8_bias_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_dense_9_kernel_v_read_readvariableop`savev2_adam_convolutional_autoencoder_4_convolution_decoder_4_dense_9_bias_v_read_readvariableopdsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_40_bias_v_read_readvariableopdsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_41_bias_v_read_readvariableopdsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_42_bias_v_read_readvariableopdsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_43_bias_v_read_readvariableopdsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_4_convolution_decoder_4_conv2d_44_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?: : : : : : ::: : : @:@:@?:?:
?	?:?:
??:?:?:?:??:?:?@:@:@ : : :: : ::: : : @:@:@?:?:
?	?:?:
??:?:?:?:??:?:?@:@:@ : : :::: : : @:@:@?:?:
?	?:?:
??:?:?:?:??:?:?@:@:@ : : :: 2(
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
:?:&"
 
_output_shapes
:
?	?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!
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
:?:&&"
 
_output_shapes
:
?	?:!'

_output_shapes	
:?:&("
 
_output_shapes
:
??:!)
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
:?:&<"
 
_output_shapes
:
?	?:!=

_output_shapes	
:?:&>"
 
_output_shapes
:
??:!?
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
?
?
D__inference_conv2d_42_layer_call_and_return_conditional_losses_60297

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
+?&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "ConvolutionalAutoencoder", "name": "convolutional_autoencoder_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"image_size": 64, "code_dim": 128, "depth": 4}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 64, 64, 1]}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ConvolutionalAutoencoder", "config": {"image_size": 64, "code_dim": 128, "depth": 4}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
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
_tf_keras_model?{"class_name": "ConvolutionEncoder", "name": "convolution_encoder_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 64, 64, 1]}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ConvolutionEncoder"}}
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
_tf_keras_model?{"class_name": "ConvolutionDecoder", "name": "convolution_decoder_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 128]}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ConvolutionDecoder"}}
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
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

&kernel
'bias
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1152}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 1152]}}
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
_tf_keras_layer?{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 128]}}
?
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_4", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4, 16]}}}
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
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_44", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_44", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 64, 64, 32]}}
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
\:Z2Bconvolutional_autoencoder_4/convolution_encoder_4/conv2d_36/kernel
N:L2@convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/bias
\:Z 2Bconvolutional_autoencoder_4/convolution_encoder_4/conv2d_37/kernel
N:L 2@convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/bias
\:Z @2Bconvolutional_autoencoder_4/convolution_encoder_4/conv2d_38/kernel
N:L@2@convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/bias
]:[@?2Bconvolutional_autoencoder_4/convolution_encoder_4/conv2d_39/kernel
O:M?2@convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/bias
T:R
?	?2@convolutional_autoencoder_4/convolution_encoder_4/dense_8/kernel
M:K?2>convolutional_autoencoder_4/convolution_encoder_4/dense_8/bias
T:R
??2@convolutional_autoencoder_4/convolution_decoder_4/dense_9/kernel
M:K?2>convolutional_autoencoder_4/convolution_decoder_4/dense_9/bias
]:[?2Bconvolutional_autoencoder_4/convolution_decoder_4/conv2d_40/kernel
O:M?2@convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/bias
^:\??2Bconvolutional_autoencoder_4/convolution_decoder_4/conv2d_41/kernel
O:M?2@convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/bias
]:[?@2Bconvolutional_autoencoder_4/convolution_decoder_4/conv2d_42/kernel
N:L@2@convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/bias
\:Z@ 2Bconvolutional_autoencoder_4/convolution_decoder_4/conv2d_43/kernel
N:L 2@convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/bias
\:Z 2Bconvolutional_autoencoder_4/convolution_decoder_4/conv2d_44/kernel
N:L2@convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/bias
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
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_36", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 64, 64, 1]}}
?	

 kernel
!bias
hregularization_losses
itrainable_variables
j	variables
k	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_37", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 31, 31, 16]}}
?	

"kernel
#bias
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_38", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 15, 15, 32]}}
?	

$kernel
%bias
pregularization_losses
qtrainable_variables
r	variables
s	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_39", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 7, 7, 64]}}
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
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_40", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 4, 4, 16]}}
?	

,kernel
-bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_41", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 8, 8, 256]}}
?	

.kernel
/bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_42", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_42", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 16, 16, 128]}}
?	

0kernel
1bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_43", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_43", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 32, 32, 64]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_16", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_17", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_18", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_19", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
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
a:_2IAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/kernel/m
S:Q2GAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/bias/m
a:_ 2IAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/kernel/m
S:Q 2GAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/bias/m
a:_ @2IAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/kernel/m
S:Q@2GAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/bias/m
b:`@?2IAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/kernel/m
T:R?2GAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/bias/m
Y:W
?	?2GAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/kernel/m
R:P?2EAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/bias/m
Y:W
??2GAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/kernel/m
R:P?2EAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/bias/m
b:`?2IAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/kernel/m
T:R?2GAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/bias/m
c:a??2IAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/kernel/m
T:R?2GAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/bias/m
b:`?@2IAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/kernel/m
S:Q@2GAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/bias/m
a:_@ 2IAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/kernel/m
S:Q 2GAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/bias/m
a:_ 2IAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/kernel/m
S:Q2GAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/bias/m
a:_2IAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/kernel/v
S:Q2GAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_36/bias/v
a:_ 2IAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/kernel/v
S:Q 2GAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_37/bias/v
a:_ @2IAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/kernel/v
S:Q@2GAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_38/bias/v
b:`@?2IAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/kernel/v
T:R?2GAdam/convolutional_autoencoder_4/convolution_encoder_4/conv2d_39/bias/v
Y:W
?	?2GAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/kernel/v
R:P?2EAdam/convolutional_autoencoder_4/convolution_encoder_4/dense_8/bias/v
Y:W
??2GAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/kernel/v
R:P?2EAdam/convolutional_autoencoder_4/convolution_decoder_4/dense_9/bias/v
b:`?2IAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/kernel/v
T:R?2GAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_40/bias/v
c:a??2IAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/kernel/v
T:R?2GAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_41/bias/v
b:`?@2IAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/kernel/v
S:Q@2GAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_42/bias/v
a:_@ 2IAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/kernel/v
S:Q 2GAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_43/bias/v
a:_ 2IAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/kernel/v
S:Q2GAdam/convolutional_autoencoder_4/convolution_decoder_4/conv2d_44/bias/v
?2?
;__inference_convolutional_autoencoder_4_layer_call_fn_60500?
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
 __inference__wrapped_model_59923?
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
V__inference_convolutional_autoencoder_4_layer_call_and_return_conditional_losses_60450?
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
5__inference_convolution_encoder_4_layer_call_fn_60102?
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
P__inference_convolution_encoder_4_layer_call_and_return_conditional_losses_60076?
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
5__inference_convolution_decoder_4_layer_call_fn_60399?
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
annotations? *'?$
"?
input_1??????????
?2?
P__inference_convolution_decoder_4_layer_call_and_return_conditional_losses_60369?
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
annotations? *'?$
"?
input_1??????????
?B?
#__inference_signature_wrapper_60559input_1"?
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
)__inference_flatten_4_layer_call_fn_60570?
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
D__inference_flatten_4_layer_call_and_return_conditional_losses_60565?
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
'__inference_dense_8_layer_call_fn_60589?
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
B__inference_dense_8_layer_call_and_return_conditional_losses_60580?
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
'__inference_dense_9_layer_call_fn_60608?
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
B__inference_dense_9_layer_call_and_return_conditional_losses_60599?
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
)__inference_reshape_4_layer_call_fn_60627?
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
D__inference_reshape_4_layer_call_and_return_conditional_losses_60622?
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
)__inference_conv2d_44_layer_call_fn_60646?
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
D__inference_conv2d_44_layer_call_and_return_conditional_losses_60637?
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
)__inference_conv2d_36_layer_call_fn_60666?
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
D__inference_conv2d_36_layer_call_and_return_conditional_losses_60657?
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
)__inference_conv2d_37_layer_call_fn_60686?
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
D__inference_conv2d_37_layer_call_and_return_conditional_losses_60677?
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
)__inference_conv2d_38_layer_call_fn_60706?
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
D__inference_conv2d_38_layer_call_and_return_conditional_losses_60697?
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
)__inference_conv2d_39_layer_call_fn_60726?
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
D__inference_conv2d_39_layer_call_and_return_conditional_losses_60717?
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
)__inference_conv2d_40_layer_call_fn_60746?
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
D__inference_conv2d_40_layer_call_and_return_conditional_losses_60737?
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
)__inference_conv2d_41_layer_call_fn_60766?
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
D__inference_conv2d_41_layer_call_and_return_conditional_losses_60757?
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
)__inference_conv2d_42_layer_call_fn_60786?
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
D__inference_conv2d_42_layer_call_and_return_conditional_losses_60777?
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
)__inference_conv2d_43_layer_call_fn_60806?
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
D__inference_conv2d_43_layer_call_and_return_conditional_losses_60797?
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
0__inference_up_sampling2d_16_layer_call_fn_60121?
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
K__inference_up_sampling2d_16_layer_call_and_return_conditional_losses_60115?
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
0__inference_up_sampling2d_17_layer_call_fn_60140?
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
K__inference_up_sampling2d_17_layer_call_and_return_conditional_losses_60134?
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
0__inference_up_sampling2d_18_layer_call_fn_60159?
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
K__inference_up_sampling2d_18_layer_call_and_return_conditional_losses_60153?
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
0__inference_up_sampling2d_19_layer_call_fn_60178?
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
K__inference_up_sampling2d_19_layer_call_and_return_conditional_losses_60172?
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
 __inference__wrapped_model_59923? !"#$%&'()*+,-./01238?5
.?+
)?&
input_1?????????@@
? ";?8
6
output_1*?'
output_1?????????@@?
D__inference_conv2d_36_layer_call_and_return_conditional_losses_60657l7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????
? ?
)__inference_conv2d_36_layer_call_fn_60666_7?4
-?*
(?%
inputs?????????@@
? " ???????????
D__inference_conv2d_37_layer_call_and_return_conditional_losses_60677l !7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
)__inference_conv2d_37_layer_call_fn_60686_ !7?4
-?*
(?%
inputs?????????
? " ?????????? ?
D__inference_conv2d_38_layer_call_and_return_conditional_losses_60697l"#7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
)__inference_conv2d_38_layer_call_fn_60706_"#7?4
-?*
(?%
inputs????????? 
? " ??????????@?
D__inference_conv2d_39_layer_call_and_return_conditional_losses_60717m$%7?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0??????????
? ?
)__inference_conv2d_39_layer_call_fn_60726`$%7?4
-?*
(?%
inputs?????????@
? "!????????????
D__inference_conv2d_40_layer_call_and_return_conditional_losses_60737m*+7?4
-?*
(?%
inputs?????????
? ".?+
$?!
0??????????
? ?
)__inference_conv2d_40_layer_call_fn_60746`*+7?4
-?*
(?%
inputs?????????
? "!????????????
D__inference_conv2d_41_layer_call_and_return_conditional_losses_60757?,-J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
)__inference_conv2d_41_layer_call_fn_60766?,-J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
D__inference_conv2d_42_layer_call_and_return_conditional_losses_60777?./J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
)__inference_conv2d_42_layer_call_fn_60786?./J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+???????????????????????????@?
D__inference_conv2d_43_layer_call_and_return_conditional_losses_60797?01I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
)__inference_conv2d_43_layer_call_fn_60806?01I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
D__inference_conv2d_44_layer_call_and_return_conditional_losses_60637?23I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
)__inference_conv2d_44_layer_call_fn_60646?23I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
P__inference_convolution_decoder_4_layer_call_and_return_conditional_losses_60369?()*+,-./01231?.
'?$
"?
input_1??????????
? "??<
5?2
0+???????????????????????????
? ?
5__inference_convolution_decoder_4_layer_call_fn_60399u()*+,-./01231?.
'?$
"?
input_1??????????
? "2?/+????????????????????????????
P__inference_convolution_encoder_4_layer_call_and_return_conditional_losses_60076n
 !"#$%&'8?5
.?+
)?&
input_1?????????@@
? "&?#
?
0??????????
? ?
5__inference_convolution_encoder_4_layer_call_fn_60102a
 !"#$%&'8?5
.?+
)?&
input_1?????????@@
? "????????????
V__inference_convolutional_autoencoder_4_layer_call_and_return_conditional_losses_60450? !"#$%&'()*+,-./01238?5
.?+
)?&
input_1?????????@@
? "??<
5?2
0+???????????????????????????
? ?
;__inference_convolutional_autoencoder_4_layer_call_fn_60500? !"#$%&'()*+,-./01238?5
.?+
)?&
input_1?????????@@
? "2?/+????????????????????????????
B__inference_dense_8_layer_call_and_return_conditional_losses_60580^&'0?-
&?#
!?
inputs??????????	
? "&?#
?
0??????????
? |
'__inference_dense_8_layer_call_fn_60589Q&'0?-
&?#
!?
inputs??????????	
? "????????????
B__inference_dense_9_layer_call_and_return_conditional_losses_60599^()0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? |
'__inference_dense_9_layer_call_fn_60608Q()0?-
&?#
!?
inputs??????????
? "????????????
D__inference_flatten_4_layer_call_and_return_conditional_losses_60565b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????	
? ?
)__inference_flatten_4_layer_call_fn_60570U8?5
.?+
)?&
inputs??????????
? "???????????	?
D__inference_reshape_4_layer_call_and_return_conditional_losses_60622a0?-
&?#
!?
inputs??????????
? "-?*
#? 
0?????????
? ?
)__inference_reshape_4_layer_call_fn_60627T0?-
&?#
!?
inputs??????????
? " ???????????
#__inference_signature_wrapper_60559? !"#$%&'()*+,-./0123C?@
? 
9?6
4
input_1)?&
input_1?????????@@";?8
6
output_1*?'
output_1?????????@@?
K__inference_up_sampling2d_16_layer_call_and_return_conditional_losses_60115?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_up_sampling2d_16_layer_call_fn_60121?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_up_sampling2d_17_layer_call_and_return_conditional_losses_60134?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_up_sampling2d_17_layer_call_fn_60140?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_up_sampling2d_18_layer_call_and_return_conditional_losses_60153?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_up_sampling2d_18_layer_call_fn_60159?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_up_sampling2d_19_layer_call_and_return_conditional_losses_60172?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_up_sampling2d_19_layer_call_fn_60178?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????