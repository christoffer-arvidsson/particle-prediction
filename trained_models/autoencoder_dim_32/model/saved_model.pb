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
Bconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*S
shared_nameDBconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/kernel
?
Vconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/kernel*&
_output_shapes
:*
dtype0
?
@convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/bias
?
Tconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/bias*
_output_shapes
:*
dtype0
?
Bconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *S
shared_nameDBconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/kernel
?
Vconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/kernel*&
_output_shapes
: *
dtype0
?
@convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/bias
?
Tconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/bias*
_output_shapes
: *
dtype0
?
Bconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*S
shared_nameDBconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/kernel
?
Vconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/kernel*&
_output_shapes
: @*
dtype0
?
@convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*Q
shared_nameB@convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/bias
?
Tconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/bias*
_output_shapes
:@*
dtype0
?
Bconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*S
shared_nameDBconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/kernel
?
Vconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/kernel*'
_output_shapes
:@?*
dtype0
?
@convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*Q
shared_nameB@convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/bias
?
Tconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/bias*
_output_shapes	
:?*
dtype0
?
@convolutional_autoencoder_2/convolution_encoder_2/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	 *Q
shared_nameB@convolutional_autoencoder_2/convolution_encoder_2/dense_4/kernel
?
Tconvolutional_autoencoder_2/convolution_encoder_2/dense_4/kernel/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_2/convolution_encoder_2/dense_4/kernel*
_output_shapes
:	?	 *
dtype0
?
>convolutional_autoencoder_2/convolution_encoder_2/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>convolutional_autoencoder_2/convolution_encoder_2/dense_4/bias
?
Rconvolutional_autoencoder_2/convolution_encoder_2/dense_4/bias/Read/ReadVariableOpReadVariableOp>convolutional_autoencoder_2/convolution_encoder_2/dense_4/bias*
_output_shapes
: *
dtype0
?
@convolutional_autoencoder_2/convolution_decoder_2/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*Q
shared_nameB@convolutional_autoencoder_2/convolution_decoder_2/dense_5/kernel
?
Tconvolutional_autoencoder_2/convolution_decoder_2/dense_5/kernel/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_2/convolution_decoder_2/dense_5/kernel*
_output_shapes
:	 ?*
dtype0
?
>convolutional_autoencoder_2/convolution_decoder_2/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*O
shared_name@>convolutional_autoencoder_2/convolution_decoder_2/dense_5/bias
?
Rconvolutional_autoencoder_2/convolution_decoder_2/dense_5/bias/Read/ReadVariableOpReadVariableOp>convolutional_autoencoder_2/convolution_decoder_2/dense_5/bias*
_output_shapes	
:?*
dtype0
?
Bconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*S
shared_nameDBconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/kernel
?
Vconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/kernel*'
_output_shapes
:?*
dtype0
?
@convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*Q
shared_nameB@convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/bias
?
Tconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/bias*
_output_shapes	
:?*
dtype0
?
Bconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*S
shared_nameDBconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/kernel
?
Vconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/kernel*(
_output_shapes
:??*
dtype0
?
@convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*Q
shared_nameB@convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/bias
?
Tconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/bias*
_output_shapes	
:?*
dtype0
?
Bconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*S
shared_nameDBconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/kernel
?
Vconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/kernel*'
_output_shapes
:?@*
dtype0
?
@convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*Q
shared_nameB@convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/bias
?
Tconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/bias*
_output_shapes
:@*
dtype0
?
Bconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *S
shared_nameDBconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/kernel
?
Vconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/kernel*&
_output_shapes
:@ *
dtype0
?
@convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/bias
?
Tconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/bias*
_output_shapes
: *
dtype0
?
Bconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *S
shared_nameDBconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/kernel
?
Vconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/kernel/Read/ReadVariableOpReadVariableOpBconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/kernel*&
_output_shapes
: *
dtype0
?
@convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/bias
?
Tconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/bias/Read/ReadVariableOpReadVariableOp@convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/bias*
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
IAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Z
shared_nameKIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/kernel/m
?
]Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/kernel/m*&
_output_shapes
:*
dtype0
?
GAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*X
shared_nameIGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/bias/m
?
[Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/bias/m*
_output_shapes
:*
dtype0
?
IAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Z
shared_nameKIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/kernel/m
?
]Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/kernel/m*&
_output_shapes
: *
dtype0
?
GAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/bias/m
?
[Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/bias/m*
_output_shapes
: *
dtype0
?
IAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*Z
shared_nameKIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/kernel/m
?
]Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/kernel/m*&
_output_shapes
: @*
dtype0
?
GAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*X
shared_nameIGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/bias/m
?
[Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/bias/m*
_output_shapes
:@*
dtype0
?
IAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*Z
shared_nameKIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/kernel/m
?
]Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/kernel/m*'
_output_shapes
:@?*
dtype0
?
GAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*X
shared_nameIGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/bias/m
?
[Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/bias/m*
_output_shapes	
:?*
dtype0
?
GAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	 *X
shared_nameIGAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/kernel/m
?
[Adam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/kernel/m*
_output_shapes
:	?	 *
dtype0
?
EAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *V
shared_nameGEAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/bias/m
?
YAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/bias/m/Read/ReadVariableOpReadVariableOpEAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/bias/m*
_output_shapes
: *
dtype0
?
GAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*X
shared_nameIGAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/kernel/m
?
[Adam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/kernel/m*
_output_shapes
:	 ?*
dtype0
?
EAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*V
shared_nameGEAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/bias/m
?
YAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/bias/m/Read/ReadVariableOpReadVariableOpEAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/bias/m*
_output_shapes	
:?*
dtype0
?
IAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*Z
shared_nameKIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/kernel/m
?
]Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/kernel/m*'
_output_shapes
:?*
dtype0
?
GAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*X
shared_nameIGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/bias/m
?
[Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/bias/m*
_output_shapes	
:?*
dtype0
?
IAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*Z
shared_nameKIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/kernel/m
?
]Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/kernel/m*(
_output_shapes
:??*
dtype0
?
GAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*X
shared_nameIGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/bias/m
?
[Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/bias/m*
_output_shapes	
:?*
dtype0
?
IAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*Z
shared_nameKIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/kernel/m
?
]Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/kernel/m*'
_output_shapes
:?@*
dtype0
?
GAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*X
shared_nameIGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/bias/m
?
[Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/bias/m*
_output_shapes
:@*
dtype0
?
IAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *Z
shared_nameKIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/kernel/m
?
]Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/kernel/m*&
_output_shapes
:@ *
dtype0
?
GAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/bias/m
?
[Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/bias/m*
_output_shapes
: *
dtype0
?
IAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Z
shared_nameKIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/kernel/m
?
]Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/kernel/m*&
_output_shapes
: *
dtype0
?
GAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*X
shared_nameIGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/bias/m
?
[Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/bias/m/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/bias/m*
_output_shapes
:*
dtype0
?
IAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Z
shared_nameKIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/kernel/v
?
]Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/kernel/v*&
_output_shapes
:*
dtype0
?
GAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*X
shared_nameIGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/bias/v
?
[Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/bias/v*
_output_shapes
:*
dtype0
?
IAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Z
shared_nameKIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/kernel/v
?
]Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/kernel/v*&
_output_shapes
: *
dtype0
?
GAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/bias/v
?
[Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/bias/v*
_output_shapes
: *
dtype0
?
IAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*Z
shared_nameKIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/kernel/v
?
]Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/kernel/v*&
_output_shapes
: @*
dtype0
?
GAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*X
shared_nameIGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/bias/v
?
[Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/bias/v*
_output_shapes
:@*
dtype0
?
IAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*Z
shared_nameKIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/kernel/v
?
]Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/kernel/v*'
_output_shapes
:@?*
dtype0
?
GAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*X
shared_nameIGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/bias/v
?
[Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/bias/v*
_output_shapes	
:?*
dtype0
?
GAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	 *X
shared_nameIGAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/kernel/v
?
[Adam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/kernel/v*
_output_shapes
:	?	 *
dtype0
?
EAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *V
shared_nameGEAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/bias/v
?
YAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/bias/v/Read/ReadVariableOpReadVariableOpEAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/bias/v*
_output_shapes
: *
dtype0
?
GAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*X
shared_nameIGAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/kernel/v
?
[Adam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/kernel/v*
_output_shapes
:	 ?*
dtype0
?
EAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*V
shared_nameGEAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/bias/v
?
YAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/bias/v/Read/ReadVariableOpReadVariableOpEAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/bias/v*
_output_shapes	
:?*
dtype0
?
IAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*Z
shared_nameKIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/kernel/v
?
]Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/kernel/v*'
_output_shapes
:?*
dtype0
?
GAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*X
shared_nameIGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/bias/v
?
[Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/bias/v*
_output_shapes	
:?*
dtype0
?
IAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*Z
shared_nameKIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/kernel/v
?
]Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/kernel/v*(
_output_shapes
:??*
dtype0
?
GAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*X
shared_nameIGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/bias/v
?
[Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/bias/v*
_output_shapes	
:?*
dtype0
?
IAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*Z
shared_nameKIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/kernel/v
?
]Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/kernel/v*'
_output_shapes
:?@*
dtype0
?
GAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*X
shared_nameIGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/bias/v
?
[Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/bias/v*
_output_shapes
:@*
dtype0
?
IAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *Z
shared_nameKIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/kernel/v
?
]Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/kernel/v*&
_output_shapes
:@ *
dtype0
?
GAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/bias/v
?
[Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/bias/v*
_output_shapes
: *
dtype0
?
IAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Z
shared_nameKIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/kernel/v
?
]Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/kernel/v*&
_output_shapes
: *
dtype0
?
GAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*X
shared_nameIGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/bias/v
?
[Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/bias/v/Read/ReadVariableOpReadVariableOpGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/bias/v*
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
VARIABLE_VALUEBconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_2/convolution_encoder_2/dense_4/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>convolutional_autoencoder_2/convolution_encoder_2/dense_4/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_2/convolution_decoder_2/dense_5/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>convolutional_autoencoder_2/convolution_decoder_2/dense_5/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/kernel1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/bias1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEEAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEEAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/kernel/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/bias/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/kernel/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/bias/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/kernel/mMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/bias/mMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/kernel/mMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/bias/mMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEEAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEEAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/kernel/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/bias/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/kernel/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/bias/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/kernel/vMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/bias/vMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/kernel/vMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/bias/vMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????@@*
dtype0*$
shape:?????????@@
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Bconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/kernel@convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/biasBconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/kernel@convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/biasBconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/kernel@convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/biasBconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/kernel@convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/bias@convolutional_autoencoder_2/convolution_encoder_2/dense_4/kernel>convolutional_autoencoder_2/convolution_encoder_2/dense_4/bias@convolutional_autoencoder_2/convolution_decoder_2/dense_5/kernel>convolutional_autoencoder_2/convolution_decoder_2/dense_5/biasBconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/kernel@convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/biasBconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/kernel@convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/biasBconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/kernel@convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/biasBconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/kernel@convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/biasBconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/kernel@convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/bias*"
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
#__inference_signature_wrapper_35777
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?3
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpVconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/kernel/Read/ReadVariableOpTconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/bias/Read/ReadVariableOpVconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/kernel/Read/ReadVariableOpTconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/bias/Read/ReadVariableOpVconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/kernel/Read/ReadVariableOpTconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/bias/Read/ReadVariableOpVconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/kernel/Read/ReadVariableOpTconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/bias/Read/ReadVariableOpTconvolutional_autoencoder_2/convolution_encoder_2/dense_4/kernel/Read/ReadVariableOpRconvolutional_autoencoder_2/convolution_encoder_2/dense_4/bias/Read/ReadVariableOpTconvolutional_autoencoder_2/convolution_decoder_2/dense_5/kernel/Read/ReadVariableOpRconvolutional_autoencoder_2/convolution_decoder_2/dense_5/bias/Read/ReadVariableOpVconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/kernel/Read/ReadVariableOpTconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/bias/Read/ReadVariableOpVconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/kernel/Read/ReadVariableOpTconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/bias/Read/ReadVariableOpVconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/kernel/Read/ReadVariableOpTconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/bias/Read/ReadVariableOpVconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/kernel/Read/ReadVariableOpTconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/bias/Read/ReadVariableOpVconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/kernel/Read/ReadVariableOpTconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp]Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/bias/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/kernel/m/Read/ReadVariableOpYAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/bias/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/kernel/m/Read/ReadVariableOpYAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/kernel/m/Read/ReadVariableOp[Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/bias/m/Read/ReadVariableOp]Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/bias/v/Read/ReadVariableOp]Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/bias/v/Read/ReadVariableOp]Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/bias/v/Read/ReadVariableOp]Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/bias/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/kernel/v/Read/ReadVariableOpYAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/bias/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/kernel/v/Read/ReadVariableOpYAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/bias/v/Read/ReadVariableOp]Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/bias/v/Read/ReadVariableOp]Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/bias/v/Read/ReadVariableOp]Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/bias/v/Read/ReadVariableOp]Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/bias/v/Read/ReadVariableOp]Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/kernel/v/Read/ReadVariableOp[Adam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/bias/v/Read/ReadVariableOpConst*V
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
__inference__traced_save_36266
?(
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateBconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/kernel@convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/biasBconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/kernel@convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/biasBconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/kernel@convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/biasBconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/kernel@convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/bias@convolutional_autoencoder_2/convolution_encoder_2/dense_4/kernel>convolutional_autoencoder_2/convolution_encoder_2/dense_4/bias@convolutional_autoencoder_2/convolution_decoder_2/dense_5/kernel>convolutional_autoencoder_2/convolution_decoder_2/dense_5/biasBconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/kernel@convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/biasBconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/kernel@convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/biasBconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/kernel@convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/biasBconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/kernel@convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/biasBconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/kernel@convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/biastotalcountIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/kernel/mGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/bias/mIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/kernel/mGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/bias/mIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/kernel/mGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/bias/mIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/kernel/mGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/bias/mGAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/kernel/mEAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/bias/mGAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/kernel/mEAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/bias/mIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/kernel/mGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/bias/mIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/kernel/mGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/bias/mIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/kernel/mGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/bias/mIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/kernel/mGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/bias/mIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/kernel/mGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/bias/mIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/kernel/vGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/bias/vIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/kernel/vGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/bias/vIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/kernel/vGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/bias/vIAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/kernel/vGAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/bias/vGAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/kernel/vEAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/bias/vGAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/kernel/vEAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/bias/vIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/kernel/vGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/bias/vIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/kernel/vGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/bias/vIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/kernel/vGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/bias/vIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/kernel/vGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/bias/vIAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/kernel/vGAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/bias/v*U
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
!__inference__traced_restore_36495??
?
`
D__inference_reshape_2_layer_call_and_return_conditional_losses_35840

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
?
5__inference_convolution_encoder_2_layer_call_fn_35320
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
:????????? *,
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
P__inference_convolution_encoder_2_layer_call_and_return_conditional_losses_352942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

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
?
~
)__inference_conv2d_24_layer_call_fn_36004

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
D__inference_conv2d_24_layer_call_and_return_conditional_losses_355152
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
?
~
)__inference_conv2d_26_layer_call_fn_35864

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
D__inference_conv2d_26_layer_call_and_return_conditional_losses_355702
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
?
D__inference_conv2d_24_layer_call_and_return_conditional_losses_35995

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
g
K__inference_up_sampling2d_11_layer_call_and_return_conditional_losses_35390

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
/__inference_up_sampling2d_9_layer_call_fn_35358

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
J__inference_up_sampling2d_9_layer_call_and_return_conditional_losses_353522
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
??
?
 __inference__wrapped_model_35141
input_1^
Zconvolutional_autoencoder_2_convolution_encoder_2_conv2d_18_conv2d_readvariableop_resource_
[convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_biasadd_readvariableop_resource^
Zconvolutional_autoencoder_2_convolution_encoder_2_conv2d_19_conv2d_readvariableop_resource_
[convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_biasadd_readvariableop_resource^
Zconvolutional_autoencoder_2_convolution_encoder_2_conv2d_20_conv2d_readvariableop_resource_
[convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_biasadd_readvariableop_resource^
Zconvolutional_autoencoder_2_convolution_encoder_2_conv2d_21_conv2d_readvariableop_resource_
[convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_biasadd_readvariableop_resource\
Xconvolutional_autoencoder_2_convolution_encoder_2_dense_4_matmul_readvariableop_resource]
Yconvolutional_autoencoder_2_convolution_encoder_2_dense_4_biasadd_readvariableop_resource\
Xconvolutional_autoencoder_2_convolution_decoder_2_dense_5_matmul_readvariableop_resource]
Yconvolutional_autoencoder_2_convolution_decoder_2_dense_5_biasadd_readvariableop_resource^
Zconvolutional_autoencoder_2_convolution_decoder_2_conv2d_22_conv2d_readvariableop_resource_
[convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_biasadd_readvariableop_resource^
Zconvolutional_autoencoder_2_convolution_decoder_2_conv2d_23_conv2d_readvariableop_resource_
[convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_biasadd_readvariableop_resource^
Zconvolutional_autoencoder_2_convolution_decoder_2_conv2d_24_conv2d_readvariableop_resource_
[convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_biasadd_readvariableop_resource^
Zconvolutional_autoencoder_2_convolution_decoder_2_conv2d_25_conv2d_readvariableop_resource_
[convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_biasadd_readvariableop_resource^
Zconvolutional_autoencoder_2_convolution_decoder_2_conv2d_26_conv2d_readvariableop_resource_
[convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_biasadd_readvariableop_resource
identity??Rconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/Conv2D/ReadVariableOp?Rconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/Conv2D/ReadVariableOp?Rconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/Conv2D/ReadVariableOp?Rconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/Conv2D/ReadVariableOp?Rconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/Conv2D/ReadVariableOp?Pconvolutional_autoencoder_2/convolution_decoder_2/dense_5/BiasAdd/ReadVariableOp?Oconvolutional_autoencoder_2/convolution_decoder_2/dense_5/MatMul/ReadVariableOp?Rconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/Conv2D/ReadVariableOp?Rconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/Conv2D/ReadVariableOp?Rconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/Conv2D/ReadVariableOp?Rconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/BiasAdd/ReadVariableOp?Qconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/Conv2D/ReadVariableOp?Pconvolutional_autoencoder_2/convolution_encoder_2/dense_4/BiasAdd/ReadVariableOp?Oconvolutional_autoencoder_2/convolution_encoder_2/dense_4/MatMul/ReadVariableOp?
Qconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_2_convolution_encoder_2_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02S
Qconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/Conv2DConv2Dinput_1Yconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2D
Bconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/Conv2D?
Rconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02T
Rconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/BiasAddBiasAddKconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/Conv2D:output:0Zconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2E
Cconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/BiasAdd?
@convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/ReluReluLconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2B
@convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/Relu?
Qconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_2_convolution_encoder_2_conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02S
Qconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/Conv2DConv2DNconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/Relu:activations:0Yconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2D
Bconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/Conv2D?
Rconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02T
Rconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/BiasAddBiasAddKconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/Conv2D:output:0Zconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2E
Cconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/BiasAdd?
@convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/ReluReluLconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2B
@convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/Relu?
Qconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_2_convolution_encoder_2_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02S
Qconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/Conv2DConv2DNconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/Relu:activations:0Yconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2D
Bconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/Conv2D?
Rconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02T
Rconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/BiasAddBiasAddKconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/Conv2D:output:0Zconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2E
Cconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/BiasAdd?
@convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/ReluReluLconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2B
@convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/Relu?
Qconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_2_convolution_encoder_2_conv2d_21_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02S
Qconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/Conv2DConv2DNconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/Relu:activations:0Yconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2D
Bconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/Conv2D?
Rconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02T
Rconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/BiasAddBiasAddKconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/Conv2D:output:0Zconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2E
Cconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/BiasAdd?
@convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/ReluReluLconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2B
@convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/Relu?
Aconvolutional_autoencoder_2/convolution_encoder_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2C
Aconvolutional_autoencoder_2/convolution_encoder_2/flatten_2/Const?
Cconvolutional_autoencoder_2/convolution_encoder_2/flatten_2/ReshapeReshapeNconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/Relu:activations:0Jconvolutional_autoencoder_2/convolution_encoder_2/flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????	2E
Cconvolutional_autoencoder_2/convolution_encoder_2/flatten_2/Reshape?
Oconvolutional_autoencoder_2/convolution_encoder_2/dense_4/MatMul/ReadVariableOpReadVariableOpXconvolutional_autoencoder_2_convolution_encoder_2_dense_4_matmul_readvariableop_resource*
_output_shapes
:	?	 *
dtype02Q
Oconvolutional_autoencoder_2/convolution_encoder_2/dense_4/MatMul/ReadVariableOp?
@convolutional_autoencoder_2/convolution_encoder_2/dense_4/MatMulMatMulLconvolutional_autoencoder_2/convolution_encoder_2/flatten_2/Reshape:output:0Wconvolutional_autoencoder_2/convolution_encoder_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2B
@convolutional_autoencoder_2/convolution_encoder_2/dense_4/MatMul?
Pconvolutional_autoencoder_2/convolution_encoder_2/dense_4/BiasAdd/ReadVariableOpReadVariableOpYconvolutional_autoencoder_2_convolution_encoder_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02R
Pconvolutional_autoencoder_2/convolution_encoder_2/dense_4/BiasAdd/ReadVariableOp?
Aconvolutional_autoencoder_2/convolution_encoder_2/dense_4/BiasAddBiasAddJconvolutional_autoencoder_2/convolution_encoder_2/dense_4/MatMul:product:0Xconvolutional_autoencoder_2/convolution_encoder_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2C
Aconvolutional_autoencoder_2/convolution_encoder_2/dense_4/BiasAdd?
Oconvolutional_autoencoder_2/convolution_decoder_2/dense_5/MatMul/ReadVariableOpReadVariableOpXconvolutional_autoencoder_2_convolution_decoder_2_dense_5_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02Q
Oconvolutional_autoencoder_2/convolution_decoder_2/dense_5/MatMul/ReadVariableOp?
@convolutional_autoencoder_2/convolution_decoder_2/dense_5/MatMulMatMulJconvolutional_autoencoder_2/convolution_encoder_2/dense_4/BiasAdd:output:0Wconvolutional_autoencoder_2/convolution_decoder_2/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2B
@convolutional_autoencoder_2/convolution_decoder_2/dense_5/MatMul?
Pconvolutional_autoencoder_2/convolution_decoder_2/dense_5/BiasAdd/ReadVariableOpReadVariableOpYconvolutional_autoencoder_2_convolution_decoder_2_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02R
Pconvolutional_autoencoder_2/convolution_decoder_2/dense_5/BiasAdd/ReadVariableOp?
Aconvolutional_autoencoder_2/convolution_decoder_2/dense_5/BiasAddBiasAddJconvolutional_autoencoder_2/convolution_decoder_2/dense_5/MatMul:product:0Xconvolutional_autoencoder_2/convolution_decoder_2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2C
Aconvolutional_autoencoder_2/convolution_decoder_2/dense_5/BiasAdd?
Aconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/ShapeShapeJconvolutional_autoencoder_2/convolution_decoder_2/dense_5/BiasAdd:output:0*
T0*
_output_shapes
:2C
Aconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/Shape?
Oconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Q
Oconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/strided_slice/stack?
Qconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2S
Qconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/strided_slice/stack_1?
Qconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
Qconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/strided_slice/stack_2?
Iconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/strided_sliceStridedSliceJconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/Shape:output:0Xconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/strided_slice/stack:output:0Zconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/strided_slice/stack_1:output:0Zconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2K
Iconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/strided_slice?
Kconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2M
Kconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/Reshape/shape/1?
Kconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2M
Kconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/Reshape/shape/2?
Kconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2M
Kconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/Reshape/shape/3?
Iconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/Reshape/shapePackRconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/strided_slice:output:0Tconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/Reshape/shape/1:output:0Tconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/Reshape/shape/2:output:0Tconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2K
Iconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/Reshape/shape?
Cconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/ReshapeReshapeJconvolutional_autoencoder_2/convolution_decoder_2/dense_5/BiasAdd:output:0Rconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2E
Cconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/Reshape?
Qconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_2_convolution_decoder_2_conv2d_22_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02S
Qconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/Conv2DConv2DLconvolutional_autoencoder_2/convolution_decoder_2/reshape_2/Reshape:output:0Yconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2D
Bconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/Conv2D?
Rconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02T
Rconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/BiasAddBiasAddKconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/Conv2D:output:0Zconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2E
Cconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/BiasAdd?
@convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/ReluReluLconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2B
@convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/Relu?
Gconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_8/ShapeShapeNconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/Relu:activations:0*
T0*
_output_shapes
:2I
Gconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_8/Shape?
Uconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2W
Uconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_8/strided_slice/stack?
Wconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Y
Wconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_8/strided_slice/stack_1?
Wconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Y
Wconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_8/strided_slice/stack_2?
Oconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_8/strided_sliceStridedSlicePconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_8/Shape:output:0^convolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_8/strided_slice/stack:output:0`convolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_8/strided_slice/stack_1:output:0`convolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2Q
Oconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_8/strided_slice?
Gconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2I
Gconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_8/Const?
Econvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_8/mulMulXconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_8/strided_slice:output:0Pconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_8/Const:output:0*
T0*
_output_shapes
:2G
Econvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_8/mul?
^convolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_8/resize/ResizeNearestNeighborResizeNearestNeighborNconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/Relu:activations:0Iconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_8/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2`
^convolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_8/resize/ResizeNearestNeighbor?
Qconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_2_convolution_decoder_2_conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02S
Qconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/Conv2DConv2Doconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_8/resize/ResizeNearestNeighbor:resized_images:0Yconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2D
Bconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/Conv2D?
Rconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02T
Rconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/BiasAddBiasAddKconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/Conv2D:output:0Zconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2E
Cconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/BiasAdd?
@convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/ReluReluLconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2B
@convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/Relu?
Gconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_9/ShapeShapeNconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/Relu:activations:0*
T0*
_output_shapes
:2I
Gconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_9/Shape?
Uconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2W
Uconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_9/strided_slice/stack?
Wconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Y
Wconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_9/strided_slice/stack_1?
Wconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Y
Wconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_9/strided_slice/stack_2?
Oconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_9/strided_sliceStridedSlicePconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_9/Shape:output:0^convolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_9/strided_slice/stack:output:0`convolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_9/strided_slice/stack_1:output:0`convolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2Q
Oconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_9/strided_slice?
Gconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2I
Gconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_9/Const?
Econvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_9/mulMulXconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_9/strided_slice:output:0Pconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_9/Const:output:0*
T0*
_output_shapes
:2G
Econvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_9/mul?
^convolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_9/resize/ResizeNearestNeighborResizeNearestNeighborNconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/Relu:activations:0Iconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_9/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2`
^convolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_9/resize/ResizeNearestNeighbor?
Qconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_2_convolution_decoder_2_conv2d_24_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02S
Qconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/Conv2DConv2Doconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_9/resize/ResizeNearestNeighbor:resized_images:0Yconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2D
Bconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/Conv2D?
Rconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02T
Rconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/BiasAddBiasAddKconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/Conv2D:output:0Zconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2E
Cconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/BiasAdd?
@convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/ReluReluLconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2B
@convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/Relu?
Hconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_10/ShapeShapeNconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/Relu:activations:0*
T0*
_output_shapes
:2J
Hconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_10/Shape?
Vconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2X
Vconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_10/strided_slice/stack?
Xconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Z
Xconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_10/strided_slice/stack_1?
Xconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Z
Xconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_10/strided_slice/stack_2?
Pconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_10/strided_sliceStridedSliceQconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_10/Shape:output:0_convolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_10/strided_slice/stack:output:0aconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_10/strided_slice/stack_1:output:0aconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2R
Pconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_10/strided_slice?
Hconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2J
Hconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_10/Const?
Fconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_10/mulMulYconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_10/strided_slice:output:0Qconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_10/Const:output:0*
T0*
_output_shapes
:2H
Fconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_10/mul?
_convolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_10/resize/ResizeNearestNeighborResizeNearestNeighborNconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/Relu:activations:0Jconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_10/mul:z:0*
T0*/
_output_shapes
:?????????  @*
half_pixel_centers(2a
_convolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_10/resize/ResizeNearestNeighbor?
Qconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_2_convolution_decoder_2_conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02S
Qconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/Conv2DConv2Dpconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_10/resize/ResizeNearestNeighbor:resized_images:0Yconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2D
Bconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/Conv2D?
Rconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02T
Rconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/BiasAddBiasAddKconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/Conv2D:output:0Zconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2E
Cconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/BiasAdd?
@convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/ReluReluLconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2B
@convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/Relu?
Hconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_11/ShapeShapeNconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/Relu:activations:0*
T0*
_output_shapes
:2J
Hconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_11/Shape?
Vconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2X
Vconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_11/strided_slice/stack?
Xconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Z
Xconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_11/strided_slice/stack_1?
Xconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Z
Xconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_11/strided_slice/stack_2?
Pconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_11/strided_sliceStridedSliceQconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_11/Shape:output:0_convolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_11/strided_slice/stack:output:0aconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_11/strided_slice/stack_1:output:0aconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2R
Pconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_11/strided_slice?
Hconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2J
Hconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_11/Const?
Fconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_11/mulMulYconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_11/strided_slice:output:0Qconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_11/Const:output:0*
T0*
_output_shapes
:2H
Fconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_11/mul?
_convolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_11/resize/ResizeNearestNeighborResizeNearestNeighborNconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/Relu:activations:0Jconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_11/mul:z:0*
T0*/
_output_shapes
:?????????@@ *
half_pixel_centers(2a
_convolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_11/resize/ResizeNearestNeighbor?
Qconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/Conv2D/ReadVariableOpReadVariableOpZconvolutional_autoencoder_2_convolution_decoder_2_conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02S
Qconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/Conv2D/ReadVariableOp?
Bconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/Conv2DConv2Dpconvolutional_autoencoder_2/convolution_decoder_2/up_sampling2d_11/resize/ResizeNearestNeighbor:resized_images:0Yconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2D
Bconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/Conv2D?
Rconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp[convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02T
Rconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/BiasAdd/ReadVariableOp?
Cconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/BiasAddBiasAddKconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/Conv2D:output:0Zconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2E
Cconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/BiasAdd?
IdentityIdentityLconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/BiasAdd:output:0S^convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/BiasAdd/ReadVariableOpR^convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/Conv2D/ReadVariableOpS^convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/BiasAdd/ReadVariableOpR^convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/Conv2D/ReadVariableOpS^convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/BiasAdd/ReadVariableOpR^convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/Conv2D/ReadVariableOpS^convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/BiasAdd/ReadVariableOpR^convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/Conv2D/ReadVariableOpS^convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/BiasAdd/ReadVariableOpR^convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/Conv2D/ReadVariableOpQ^convolutional_autoencoder_2/convolution_decoder_2/dense_5/BiasAdd/ReadVariableOpP^convolutional_autoencoder_2/convolution_decoder_2/dense_5/MatMul/ReadVariableOpS^convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/BiasAdd/ReadVariableOpR^convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/Conv2D/ReadVariableOpS^convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/BiasAdd/ReadVariableOpR^convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/Conv2D/ReadVariableOpS^convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/BiasAdd/ReadVariableOpR^convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/Conv2D/ReadVariableOpS^convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/BiasAdd/ReadVariableOpR^convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/Conv2D/ReadVariableOpQ^convolutional_autoencoder_2/convolution_encoder_2/dense_4/BiasAdd/ReadVariableOpP^convolutional_autoencoder_2/convolution_encoder_2/dense_4/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????@@::::::::::::::::::::::2?
Rconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/BiasAdd/ReadVariableOpRconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/Conv2D/ReadVariableOpQconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/Conv2D/ReadVariableOp2?
Rconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/BiasAdd/ReadVariableOpRconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/Conv2D/ReadVariableOpQconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/Conv2D/ReadVariableOp2?
Rconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/BiasAdd/ReadVariableOpRconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/Conv2D/ReadVariableOpQconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/Conv2D/ReadVariableOp2?
Rconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/BiasAdd/ReadVariableOpRconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/Conv2D/ReadVariableOpQconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/Conv2D/ReadVariableOp2?
Rconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/BiasAdd/ReadVariableOpRconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/Conv2D/ReadVariableOpQconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/Conv2D/ReadVariableOp2?
Pconvolutional_autoencoder_2/convolution_decoder_2/dense_5/BiasAdd/ReadVariableOpPconvolutional_autoencoder_2/convolution_decoder_2/dense_5/BiasAdd/ReadVariableOp2?
Oconvolutional_autoencoder_2/convolution_decoder_2/dense_5/MatMul/ReadVariableOpOconvolutional_autoencoder_2/convolution_decoder_2/dense_5/MatMul/ReadVariableOp2?
Rconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/BiasAdd/ReadVariableOpRconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/Conv2D/ReadVariableOpQconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/Conv2D/ReadVariableOp2?
Rconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/BiasAdd/ReadVariableOpRconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/Conv2D/ReadVariableOpQconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/Conv2D/ReadVariableOp2?
Rconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/BiasAdd/ReadVariableOpRconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/Conv2D/ReadVariableOpQconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/Conv2D/ReadVariableOp2?
Rconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/BiasAdd/ReadVariableOpRconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/BiasAdd/ReadVariableOp2?
Qconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/Conv2D/ReadVariableOpQconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/Conv2D/ReadVariableOp2?
Pconvolutional_autoencoder_2/convolution_encoder_2/dense_4/BiasAdd/ReadVariableOpPconvolutional_autoencoder_2/convolution_encoder_2/dense_4/BiasAdd/ReadVariableOp2?
Oconvolutional_autoencoder_2/convolution_encoder_2/dense_4/MatMul/ReadVariableOpOconvolutional_autoencoder_2/convolution_encoder_2/dense_4/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?
`
D__inference_reshape_2_layer_call_and_return_conditional_losses_35440

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
?
E
)__inference_flatten_2_layer_call_fn_35788

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
D__inference_flatten_2_layer_call_and_return_conditional_losses_352592
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
|
'__inference_dense_5_layer_call_fn_35826

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
B__inference_dense_5_layer_call_and_return_conditional_losses_354102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
D__inference_conv2d_20_layer_call_and_return_conditional_losses_35210

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
f
J__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_35333

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
?
P__inference_convolution_encoder_2_layer_call_and_return_conditional_losses_35294
input_1
conv2d_18_35167
conv2d_18_35169
conv2d_19_35194
conv2d_19_35196
conv2d_20_35221
conv2d_20_35223
conv2d_21_35248
conv2d_21_35250
dense_4_35288
dense_4_35290
identity??!conv2d_18/StatefulPartitionedCall?!conv2d_19/StatefulPartitionedCall?!conv2d_20/StatefulPartitionedCall?!conv2d_21/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_18_35167conv2d_18_35169*
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
D__inference_conv2d_18_layer_call_and_return_conditional_losses_351562#
!conv2d_18/StatefulPartitionedCall?
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0conv2d_19_35194conv2d_19_35196*
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
D__inference_conv2d_19_layer_call_and_return_conditional_losses_351832#
!conv2d_19/StatefulPartitionedCall?
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0conv2d_20_35221conv2d_20_35223*
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
D__inference_conv2d_20_layer_call_and_return_conditional_losses_352102#
!conv2d_20/StatefulPartitionedCall?
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0conv2d_21_35248conv2d_21_35250*
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
D__inference_conv2d_21_layer_call_and_return_conditional_losses_352372#
!conv2d_21/StatefulPartitionedCall?
flatten_2/PartitionedCallPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0*
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
D__inference_flatten_2_layer_call_and_return_conditional_losses_352592
flatten_2/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_35288dense_4_35290*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_352772!
dense_4/StatefulPartitionedCall?
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????@@::::::::::2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?
~
)__inference_conv2d_18_layer_call_fn_35884

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
D__inference_conv2d_18_layer_call_and_return_conditional_losses_351562
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
D__inference_conv2d_24_layer_call_and_return_conditional_losses_35515

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
)__inference_conv2d_21_layer_call_fn_35944

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
D__inference_conv2d_21_layer_call_and_return_conditional_losses_352372
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
D__inference_conv2d_26_layer_call_and_return_conditional_losses_35855

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
L
0__inference_up_sampling2d_11_layer_call_fn_35396

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
K__inference_up_sampling2d_11_layer_call_and_return_conditional_losses_353902
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
D__inference_conv2d_18_layer_call_and_return_conditional_losses_35875

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
?
#__inference_signature_wrapper_35777
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
 __inference__wrapped_model_351412
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
'__inference_dense_4_layer_call_fn_35807

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
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_352772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

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
~
)__inference_conv2d_20_layer_call_fn_35924

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
D__inference_conv2d_20_layer_call_and_return_conditional_losses_352102
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
?
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_35259

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
D__inference_conv2d_18_layer_call_and_return_conditional_losses_35156

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
5__inference_convolution_decoder_2_layer_call_fn_35617
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
P__inference_convolution_decoder_2_layer_call_and_return_conditional_losses_355872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:????????? ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:????????? 
!
_user_specified_name	input_1
?	
?
B__inference_dense_5_layer_call_and_return_conditional_losses_35410

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ?*
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
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
D__inference_conv2d_23_layer_call_and_return_conditional_losses_35975

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
;__inference_convolutional_autoencoder_2_layer_call_fn_35718
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
V__inference_convolutional_autoencoder_2_layer_call_and_return_conditional_losses_356682
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
?1
?
P__inference_convolution_decoder_2_layer_call_and_return_conditional_losses_35587
input_1
dense_5_35421
dense_5_35423
conv2d_22_35470
conv2d_22_35472
conv2d_23_35498
conv2d_23_35500
conv2d_24_35526
conv2d_24_35528
conv2d_25_35554
conv2d_25_35556
conv2d_26_35581
conv2d_26_35583
identity??!conv2d_22/StatefulPartitionedCall?!conv2d_23/StatefulPartitionedCall?!conv2d_24/StatefulPartitionedCall?!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_5_35421dense_5_35423*
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
B__inference_dense_5_layer_call_and_return_conditional_losses_354102!
dense_5/StatefulPartitionedCall?
reshape_2/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
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
D__inference_reshape_2_layer_call_and_return_conditional_losses_354402
reshape_2/PartitionedCall?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0conv2d_22_35470conv2d_22_35472*
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
D__inference_conv2d_22_layer_call_and_return_conditional_losses_354592#
!conv2d_22/StatefulPartitionedCall?
up_sampling2d_8/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
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
J__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_353332!
up_sampling2d_8/PartitionedCall?
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_8/PartitionedCall:output:0conv2d_23_35498conv2d_23_35500*
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
D__inference_conv2d_23_layer_call_and_return_conditional_losses_354872#
!conv2d_23/StatefulPartitionedCall?
up_sampling2d_9/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
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
J__inference_up_sampling2d_9_layer_call_and_return_conditional_losses_353522!
up_sampling2d_9/PartitionedCall?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_9/PartitionedCall:output:0conv2d_24_35526conv2d_24_35528*
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
D__inference_conv2d_24_layer_call_and_return_conditional_losses_355152#
!conv2d_24/StatefulPartitionedCall?
 up_sampling2d_10/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
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
K__inference_up_sampling2d_10_layer_call_and_return_conditional_losses_353712"
 up_sampling2d_10/PartitionedCall?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_10/PartitionedCall:output:0conv2d_25_35554conv2d_25_35556*
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
D__inference_conv2d_25_layer_call_and_return_conditional_losses_355432#
!conv2d_25/StatefulPartitionedCall?
 up_sampling2d_11/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
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
K__inference_up_sampling2d_11_layer_call_and_return_conditional_losses_353902"
 up_sampling2d_11/PartitionedCall?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_11/PartitionedCall:output:0conv2d_26_35581conv2d_26_35583*
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
D__inference_conv2d_26_layer_call_and_return_conditional_losses_355702#
!conv2d_26/StatefulPartitionedCall?
IdentityIdentity*conv2d_26/StatefulPartitionedCall:output:0"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:????????? ::::::::::::2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:P L
'
_output_shapes
:????????? 
!
_user_specified_name	input_1
?

?
D__inference_conv2d_22_layer_call_and_return_conditional_losses_35955

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
~
)__inference_conv2d_22_layer_call_fn_35964

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
D__inference_conv2d_22_layer_call_and_return_conditional_losses_354592
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
?
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_35783

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
D__inference_conv2d_20_layer_call_and_return_conditional_losses_35915

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
)__inference_conv2d_23_layer_call_fn_35984

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
D__inference_conv2d_23_layer_call_and_return_conditional_losses_354872
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

?
D__inference_conv2d_21_layer_call_and_return_conditional_losses_35935

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
B__inference_dense_5_layer_call_and_return_conditional_losses_35817

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ?*
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
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
D__inference_conv2d_25_layer_call_and_return_conditional_losses_36015

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
??
?7
__inference__traced_save_36266
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopa
]savev2_convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_bias_read_readvariableopa
]savev2_convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_bias_read_readvariableopa
]savev2_convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_bias_read_readvariableopa
]savev2_convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_bias_read_readvariableop_
[savev2_convolutional_autoencoder_2_convolution_encoder_2_dense_4_kernel_read_readvariableop]
Ysavev2_convolutional_autoencoder_2_convolution_encoder_2_dense_4_bias_read_readvariableop_
[savev2_convolutional_autoencoder_2_convolution_decoder_2_dense_5_kernel_read_readvariableop]
Ysavev2_convolutional_autoencoder_2_convolution_decoder_2_dense_5_bias_read_readvariableopa
]savev2_convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_bias_read_readvariableopa
]savev2_convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_bias_read_readvariableopa
]savev2_convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_bias_read_readvariableopa
]savev2_convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_bias_read_readvariableopa
]savev2_convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_kernel_read_readvariableop_
[savev2_convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_bias_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_dense_4_kernel_m_read_readvariableopd
`savev2_adam_convolutional_autoencoder_2_convolution_encoder_2_dense_4_bias_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_dense_5_kernel_m_read_readvariableopd
`savev2_adam_convolutional_autoencoder_2_convolution_decoder_2_dense_5_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_kernel_m_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_bias_m_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_bias_v_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_bias_v_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_bias_v_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_bias_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_dense_4_kernel_v_read_readvariableopd
`savev2_adam_convolutional_autoencoder_2_convolution_encoder_2_dense_4_bias_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_dense_5_kernel_v_read_readvariableopd
`savev2_adam_convolutional_autoencoder_2_convolution_decoder_2_dense_5_bias_v_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_bias_v_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_bias_v_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_bias_v_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_bias_v_read_readvariableoph
dsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_kernel_v_read_readvariableopf
bsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop]savev2_convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_kernel_read_readvariableop[savev2_convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_bias_read_readvariableop]savev2_convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_kernel_read_readvariableop[savev2_convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_bias_read_readvariableop]savev2_convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_kernel_read_readvariableop[savev2_convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_bias_read_readvariableop]savev2_convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_kernel_read_readvariableop[savev2_convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_bias_read_readvariableop[savev2_convolutional_autoencoder_2_convolution_encoder_2_dense_4_kernel_read_readvariableopYsavev2_convolutional_autoencoder_2_convolution_encoder_2_dense_4_bias_read_readvariableop[savev2_convolutional_autoencoder_2_convolution_decoder_2_dense_5_kernel_read_readvariableopYsavev2_convolutional_autoencoder_2_convolution_decoder_2_dense_5_bias_read_readvariableop]savev2_convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_kernel_read_readvariableop[savev2_convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_bias_read_readvariableop]savev2_convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_kernel_read_readvariableop[savev2_convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_bias_read_readvariableop]savev2_convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_kernel_read_readvariableop[savev2_convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_bias_read_readvariableop]savev2_convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_kernel_read_readvariableop[savev2_convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_bias_read_readvariableop]savev2_convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_kernel_read_readvariableop[savev2_convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopdsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_bias_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_dense_4_kernel_m_read_readvariableop`savev2_adam_convolutional_autoencoder_2_convolution_encoder_2_dense_4_bias_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_dense_5_kernel_m_read_readvariableop`savev2_adam_convolutional_autoencoder_2_convolution_decoder_2_dense_5_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_kernel_m_read_readvariableopbsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_bias_m_read_readvariableopdsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_bias_v_read_readvariableopdsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_bias_v_read_readvariableopdsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_bias_v_read_readvariableopdsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_bias_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_2_convolution_encoder_2_dense_4_kernel_v_read_readvariableop`savev2_adam_convolutional_autoencoder_2_convolution_encoder_2_dense_4_bias_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_dense_5_kernel_v_read_readvariableop`savev2_adam_convolutional_autoencoder_2_convolution_decoder_2_dense_5_bias_v_read_readvariableopdsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_bias_v_read_readvariableopdsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_bias_v_read_readvariableopdsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_bias_v_read_readvariableopdsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_bias_v_read_readvariableopdsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_kernel_v_read_readvariableopbsavev2_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?: : : : : : ::: : : @:@:@?:?:	?	 : :	 ?:?:?:?:??:?:?@:@:@ : : :: : ::: : : @:@:@?:?:	?	 : :	 ?:?:?:?:??:?:?@:@:@ : : :::: : : @:@:@?:?:	?	 : :	 ?:?:?:?:??:?:?@:@:@ : : :: 2(
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
:	?	 : 

_output_shapes
: :%!

_output_shapes
:	 ?:!
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
:	?	 : '

_output_shapes
: :%(!

_output_shapes
:	 ?:!)
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
:	?	 : =

_output_shapes
: :%>!

_output_shapes
:	 ?:!?
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
?
?
V__inference_convolutional_autoencoder_2_layer_call_and_return_conditional_losses_35668
input_1
convolution_encoder_2_35621
convolution_encoder_2_35623
convolution_encoder_2_35625
convolution_encoder_2_35627
convolution_encoder_2_35629
convolution_encoder_2_35631
convolution_encoder_2_35633
convolution_encoder_2_35635
convolution_encoder_2_35637
convolution_encoder_2_35639
convolution_decoder_2_35642
convolution_decoder_2_35644
convolution_decoder_2_35646
convolution_decoder_2_35648
convolution_decoder_2_35650
convolution_decoder_2_35652
convolution_decoder_2_35654
convolution_decoder_2_35656
convolution_decoder_2_35658
convolution_decoder_2_35660
convolution_decoder_2_35662
convolution_decoder_2_35664
identity??-convolution_decoder_2/StatefulPartitionedCall?-convolution_encoder_2/StatefulPartitionedCall?
-convolution_encoder_2/StatefulPartitionedCallStatefulPartitionedCallinput_1convolution_encoder_2_35621convolution_encoder_2_35623convolution_encoder_2_35625convolution_encoder_2_35627convolution_encoder_2_35629convolution_encoder_2_35631convolution_encoder_2_35633convolution_encoder_2_35635convolution_encoder_2_35637convolution_encoder_2_35639*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *,
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
P__inference_convolution_encoder_2_layer_call_and_return_conditional_losses_352942/
-convolution_encoder_2/StatefulPartitionedCall?
-convolution_decoder_2/StatefulPartitionedCallStatefulPartitionedCall6convolution_encoder_2/StatefulPartitionedCall:output:0convolution_decoder_2_35642convolution_decoder_2_35644convolution_decoder_2_35646convolution_decoder_2_35648convolution_decoder_2_35650convolution_decoder_2_35652convolution_decoder_2_35654convolution_decoder_2_35656convolution_decoder_2_35658convolution_decoder_2_35660convolution_decoder_2_35662convolution_decoder_2_35664*
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
P__inference_convolution_decoder_2_layer_call_and_return_conditional_losses_355872/
-convolution_decoder_2/StatefulPartitionedCall?
IdentityIdentity6convolution_decoder_2/StatefulPartitionedCall:output:0.^convolution_decoder_2/StatefulPartitionedCall.^convolution_encoder_2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????@@::::::::::::::::::::::2^
-convolution_decoder_2/StatefulPartitionedCall-convolution_decoder_2/StatefulPartitionedCall2^
-convolution_encoder_2/StatefulPartitionedCall-convolution_encoder_2/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?	
?
B__inference_dense_4_layer_call_and_return_conditional_losses_35277

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?	 *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

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
D__inference_conv2d_22_layer_call_and_return_conditional_losses_35459

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
D__inference_conv2d_21_layer_call_and_return_conditional_losses_35237

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
)__inference_conv2d_25_layer_call_fn_36024

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
D__inference_conv2d_25_layer_call_and_return_conditional_losses_355432
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

?
D__inference_conv2d_19_layer_call_and_return_conditional_losses_35183

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
f
J__inference_up_sampling2d_9_layer_call_and_return_conditional_losses_35352

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
B__inference_dense_4_layer_call_and_return_conditional_losses_35798

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?	 *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

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
L
0__inference_up_sampling2d_10_layer_call_fn_35377

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
K__inference_up_sampling2d_10_layer_call_and_return_conditional_losses_353712
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
D__inference_conv2d_25_layer_call_and_return_conditional_losses_35543

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
~
)__inference_conv2d_19_layer_call_fn_35904

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
D__inference_conv2d_19_layer_call_and_return_conditional_losses_351832
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
?
E
)__inference_reshape_2_layer_call_fn_35845

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
D__inference_reshape_2_layer_call_and_return_conditional_losses_354402
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
D__inference_conv2d_26_layer_call_and_return_conditional_losses_35570

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
K__inference_up_sampling2d_10_layer_call_and_return_conditional_losses_35371

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
D__inference_conv2d_23_layer_call_and_return_conditional_losses_35487

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
K
/__inference_up_sampling2d_8_layer_call_fn_35339

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
J__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_353332
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
D__inference_conv2d_19_layer_call_and_return_conditional_losses_35895

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
??
?@
!__inference__traced_restore_36495
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rateY
Uassignvariableop_5_convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_kernelW
Sassignvariableop_6_convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_biasY
Uassignvariableop_7_convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_kernelW
Sassignvariableop_8_convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_biasY
Uassignvariableop_9_convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_kernelX
Tassignvariableop_10_convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_biasZ
Vassignvariableop_11_convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_kernelX
Tassignvariableop_12_convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_biasX
Tassignvariableop_13_convolutional_autoencoder_2_convolution_encoder_2_dense_4_kernelV
Rassignvariableop_14_convolutional_autoencoder_2_convolution_encoder_2_dense_4_biasX
Tassignvariableop_15_convolutional_autoencoder_2_convolution_decoder_2_dense_5_kernelV
Rassignvariableop_16_convolutional_autoencoder_2_convolution_decoder_2_dense_5_biasZ
Vassignvariableop_17_convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_kernelX
Tassignvariableop_18_convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_biasZ
Vassignvariableop_19_convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_kernelX
Tassignvariableop_20_convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_biasZ
Vassignvariableop_21_convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_kernelX
Tassignvariableop_22_convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_biasZ
Vassignvariableop_23_convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_kernelX
Tassignvariableop_24_convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_biasZ
Vassignvariableop_25_convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_kernelX
Tassignvariableop_26_convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_bias
assignvariableop_27_total
assignvariableop_28_counta
]assignvariableop_29_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_kernel_m_
[assignvariableop_30_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_bias_ma
]assignvariableop_31_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_kernel_m_
[assignvariableop_32_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_bias_ma
]assignvariableop_33_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_kernel_m_
[assignvariableop_34_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_bias_ma
]assignvariableop_35_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_kernel_m_
[assignvariableop_36_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_bias_m_
[assignvariableop_37_adam_convolutional_autoencoder_2_convolution_encoder_2_dense_4_kernel_m]
Yassignvariableop_38_adam_convolutional_autoencoder_2_convolution_encoder_2_dense_4_bias_m_
[assignvariableop_39_adam_convolutional_autoencoder_2_convolution_decoder_2_dense_5_kernel_m]
Yassignvariableop_40_adam_convolutional_autoencoder_2_convolution_decoder_2_dense_5_bias_ma
]assignvariableop_41_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_kernel_m_
[assignvariableop_42_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_bias_ma
]assignvariableop_43_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_kernel_m_
[assignvariableop_44_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_bias_ma
]assignvariableop_45_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_kernel_m_
[assignvariableop_46_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_bias_ma
]assignvariableop_47_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_kernel_m_
[assignvariableop_48_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_bias_ma
]assignvariableop_49_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_kernel_m_
[assignvariableop_50_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_bias_ma
]assignvariableop_51_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_kernel_v_
[assignvariableop_52_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_bias_va
]assignvariableop_53_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_kernel_v_
[assignvariableop_54_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_bias_va
]assignvariableop_55_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_kernel_v_
[assignvariableop_56_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_bias_va
]assignvariableop_57_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_kernel_v_
[assignvariableop_58_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_bias_v_
[assignvariableop_59_adam_convolutional_autoencoder_2_convolution_encoder_2_dense_4_kernel_v]
Yassignvariableop_60_adam_convolutional_autoencoder_2_convolution_encoder_2_dense_4_bias_v_
[assignvariableop_61_adam_convolutional_autoencoder_2_convolution_decoder_2_dense_5_kernel_v]
Yassignvariableop_62_adam_convolutional_autoencoder_2_convolution_decoder_2_dense_5_bias_va
]assignvariableop_63_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_kernel_v_
[assignvariableop_64_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_bias_va
]assignvariableop_65_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_kernel_v_
[assignvariableop_66_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_bias_va
]assignvariableop_67_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_kernel_v_
[assignvariableop_68_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_bias_va
]assignvariableop_69_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_kernel_v_
[assignvariableop_70_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_bias_va
]assignvariableop_71_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_kernel_v_
[assignvariableop_72_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_bias_v
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
AssignVariableOp_5AssignVariableOpUassignvariableop_5_convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpSassignvariableop_6_convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpUassignvariableop_7_convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpSassignvariableop_8_convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpUassignvariableop_9_convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpTassignvariableop_10_convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpVassignvariableop_11_convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpTassignvariableop_12_convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpTassignvariableop_13_convolutional_autoencoder_2_convolution_encoder_2_dense_4_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpRassignvariableop_14_convolutional_autoencoder_2_convolution_encoder_2_dense_4_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpTassignvariableop_15_convolutional_autoencoder_2_convolution_decoder_2_dense_5_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpRassignvariableop_16_convolutional_autoencoder_2_convolution_decoder_2_dense_5_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpVassignvariableop_17_convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpTassignvariableop_18_convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpVassignvariableop_19_convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpTassignvariableop_20_convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpVassignvariableop_21_convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpTassignvariableop_22_convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpVassignvariableop_23_convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpTassignvariableop_24_convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpVassignvariableop_25_convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpTassignvariableop_26_convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_biasIdentity_26:output:0"/device:CPU:0*
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
AssignVariableOp_29AssignVariableOp]assignvariableop_29_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp[assignvariableop_30_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp]assignvariableop_31_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp[assignvariableop_32_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp]assignvariableop_33_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp[assignvariableop_34_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp]assignvariableop_35_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp[assignvariableop_36_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp[assignvariableop_37_adam_convolutional_autoencoder_2_convolution_encoder_2_dense_4_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpYassignvariableop_38_adam_convolutional_autoencoder_2_convolution_encoder_2_dense_4_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp[assignvariableop_39_adam_convolutional_autoencoder_2_convolution_decoder_2_dense_5_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpYassignvariableop_40_adam_convolutional_autoencoder_2_convolution_decoder_2_dense_5_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp]assignvariableop_41_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp[assignvariableop_42_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp]assignvariableop_43_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp[assignvariableop_44_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp]assignvariableop_45_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp[assignvariableop_46_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp]assignvariableop_47_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp[assignvariableop_48_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp]assignvariableop_49_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp[assignvariableop_50_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp]assignvariableop_51_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp[assignvariableop_52_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_18_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp]assignvariableop_53_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp[assignvariableop_54_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_19_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp]assignvariableop_55_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp[assignvariableop_56_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_20_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp]assignvariableop_57_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp[assignvariableop_58_adam_convolutional_autoencoder_2_convolution_encoder_2_conv2d_21_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp[assignvariableop_59_adam_convolutional_autoencoder_2_convolution_encoder_2_dense_4_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOpYassignvariableop_60_adam_convolutional_autoencoder_2_convolution_encoder_2_dense_4_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp[assignvariableop_61_adam_convolutional_autoencoder_2_convolution_decoder_2_dense_5_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOpYassignvariableop_62_adam_convolutional_autoencoder_2_convolution_decoder_2_dense_5_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp]assignvariableop_63_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp[assignvariableop_64_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_22_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp]assignvariableop_65_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp[assignvariableop_66_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_23_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp]assignvariableop_67_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp[assignvariableop_68_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_24_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp]assignvariableop_69_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp[assignvariableop_70_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_25_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp]assignvariableop_71_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp[assignvariableop_72_adam_convolutional_autoencoder_2_convolution_decoder_2_conv2d_26_bias_vIdentity_72:output:0"/device:CPU:0*
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
_user_specified_namefile_prefix"?L
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
_tf_keras_model?{"class_name": "ConvolutionalAutoencoder", "name": "convolutional_autoencoder_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"image_size": 64, "code_dim": 32, "depth": 4}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 64, 64, 1]}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ConvolutionalAutoencoder", "config": {"image_size": 64, "code_dim": 32, "depth": 4}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
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
_tf_keras_model?{"class_name": "ConvolutionEncoder", "name": "convolution_encoder_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 64, 64, 1]}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ConvolutionEncoder"}}
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
_tf_keras_model?{"class_name": "ConvolutionDecoder", "name": "convolution_decoder_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 32]}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ConvolutionDecoder"}}
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
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

&kernel
'bias
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1152}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 1152]}}
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
_tf_keras_layer?{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 32]}}
?
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4, 16]}}}
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
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 64, 64, 32]}}
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
\:Z2Bconvolutional_autoencoder_2/convolution_encoder_2/conv2d_18/kernel
N:L2@convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/bias
\:Z 2Bconvolutional_autoencoder_2/convolution_encoder_2/conv2d_19/kernel
N:L 2@convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/bias
\:Z @2Bconvolutional_autoencoder_2/convolution_encoder_2/conv2d_20/kernel
N:L@2@convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/bias
]:[@?2Bconvolutional_autoencoder_2/convolution_encoder_2/conv2d_21/kernel
O:M?2@convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/bias
S:Q	?	 2@convolutional_autoencoder_2/convolution_encoder_2/dense_4/kernel
L:J 2>convolutional_autoencoder_2/convolution_encoder_2/dense_4/bias
S:Q	 ?2@convolutional_autoencoder_2/convolution_decoder_2/dense_5/kernel
M:K?2>convolutional_autoencoder_2/convolution_decoder_2/dense_5/bias
]:[?2Bconvolutional_autoencoder_2/convolution_decoder_2/conv2d_22/kernel
O:M?2@convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/bias
^:\??2Bconvolutional_autoencoder_2/convolution_decoder_2/conv2d_23/kernel
O:M?2@convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/bias
]:[?@2Bconvolutional_autoencoder_2/convolution_decoder_2/conv2d_24/kernel
N:L@2@convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/bias
\:Z@ 2Bconvolutional_autoencoder_2/convolution_decoder_2/conv2d_25/kernel
N:L 2@convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/bias
\:Z 2Bconvolutional_autoencoder_2/convolution_decoder_2/conv2d_26/kernel
N:L2@convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/bias
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
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_18", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 64, 64, 1]}}
?	

 kernel
!bias
hregularization_losses
itrainable_variables
j	variables
k	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 31, 31, 16]}}
?	

"kernel
#bias
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 15, 15, 32]}}
?	

$kernel
%bias
pregularization_losses
qtrainable_variables
r	variables
s	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_21", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 7, 7, 64]}}
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
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_22", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 4, 4, 16]}}
?	

,kernel
-bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_23", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 8, 8, 256]}}
?	

.kernel
/bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 16, 16, 128]}}
?	

0kernel
1bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 32, 32, 64]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_8", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_9", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_10", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_11", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
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
a:_2IAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/kernel/m
S:Q2GAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/bias/m
a:_ 2IAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/kernel/m
S:Q 2GAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/bias/m
a:_ @2IAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/kernel/m
S:Q@2GAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/bias/m
b:`@?2IAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/kernel/m
T:R?2GAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/bias/m
X:V	?	 2GAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/kernel/m
Q:O 2EAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/bias/m
X:V	 ?2GAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/kernel/m
R:P?2EAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/bias/m
b:`?2IAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/kernel/m
T:R?2GAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/bias/m
c:a??2IAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/kernel/m
T:R?2GAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/bias/m
b:`?@2IAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/kernel/m
S:Q@2GAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/bias/m
a:_@ 2IAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/kernel/m
S:Q 2GAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/bias/m
a:_ 2IAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/kernel/m
S:Q2GAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/bias/m
a:_2IAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/kernel/v
S:Q2GAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_18/bias/v
a:_ 2IAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/kernel/v
S:Q 2GAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_19/bias/v
a:_ @2IAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/kernel/v
S:Q@2GAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_20/bias/v
b:`@?2IAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/kernel/v
T:R?2GAdam/convolutional_autoencoder_2/convolution_encoder_2/conv2d_21/bias/v
X:V	?	 2GAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/kernel/v
Q:O 2EAdam/convolutional_autoencoder_2/convolution_encoder_2/dense_4/bias/v
X:V	 ?2GAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/kernel/v
R:P?2EAdam/convolutional_autoencoder_2/convolution_decoder_2/dense_5/bias/v
b:`?2IAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/kernel/v
T:R?2GAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_22/bias/v
c:a??2IAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/kernel/v
T:R?2GAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_23/bias/v
b:`?@2IAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/kernel/v
S:Q@2GAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_24/bias/v
a:_@ 2IAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/kernel/v
S:Q 2GAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_25/bias/v
a:_ 2IAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/kernel/v
S:Q2GAdam/convolutional_autoencoder_2/convolution_decoder_2/conv2d_26/bias/v
?2?
;__inference_convolutional_autoencoder_2_layer_call_fn_35718?
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
 __inference__wrapped_model_35141?
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
V__inference_convolutional_autoencoder_2_layer_call_and_return_conditional_losses_35668?
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
5__inference_convolution_encoder_2_layer_call_fn_35320?
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
P__inference_convolution_encoder_2_layer_call_and_return_conditional_losses_35294?
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
5__inference_convolution_decoder_2_layer_call_fn_35617?
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
input_1????????? 
?2?
P__inference_convolution_decoder_2_layer_call_and_return_conditional_losses_35587?
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
input_1????????? 
?B?
#__inference_signature_wrapper_35777input_1"?
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
)__inference_flatten_2_layer_call_fn_35788?
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
D__inference_flatten_2_layer_call_and_return_conditional_losses_35783?
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
'__inference_dense_4_layer_call_fn_35807?
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
B__inference_dense_4_layer_call_and_return_conditional_losses_35798?
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
'__inference_dense_5_layer_call_fn_35826?
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
B__inference_dense_5_layer_call_and_return_conditional_losses_35817?
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
)__inference_reshape_2_layer_call_fn_35845?
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
D__inference_reshape_2_layer_call_and_return_conditional_losses_35840?
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
)__inference_conv2d_26_layer_call_fn_35864?
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
D__inference_conv2d_26_layer_call_and_return_conditional_losses_35855?
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
)__inference_conv2d_18_layer_call_fn_35884?
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
D__inference_conv2d_18_layer_call_and_return_conditional_losses_35875?
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
)__inference_conv2d_19_layer_call_fn_35904?
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
D__inference_conv2d_19_layer_call_and_return_conditional_losses_35895?
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
)__inference_conv2d_20_layer_call_fn_35924?
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
D__inference_conv2d_20_layer_call_and_return_conditional_losses_35915?
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
)__inference_conv2d_21_layer_call_fn_35944?
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
D__inference_conv2d_21_layer_call_and_return_conditional_losses_35935?
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
)__inference_conv2d_22_layer_call_fn_35964?
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
D__inference_conv2d_22_layer_call_and_return_conditional_losses_35955?
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
)__inference_conv2d_23_layer_call_fn_35984?
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
D__inference_conv2d_23_layer_call_and_return_conditional_losses_35975?
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
)__inference_conv2d_24_layer_call_fn_36004?
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
D__inference_conv2d_24_layer_call_and_return_conditional_losses_35995?
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
)__inference_conv2d_25_layer_call_fn_36024?
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
D__inference_conv2d_25_layer_call_and_return_conditional_losses_36015?
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
/__inference_up_sampling2d_8_layer_call_fn_35339?
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
J__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_35333?
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
/__inference_up_sampling2d_9_layer_call_fn_35358?
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
J__inference_up_sampling2d_9_layer_call_and_return_conditional_losses_35352?
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
0__inference_up_sampling2d_10_layer_call_fn_35377?
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
K__inference_up_sampling2d_10_layer_call_and_return_conditional_losses_35371?
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
0__inference_up_sampling2d_11_layer_call_fn_35396?
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
K__inference_up_sampling2d_11_layer_call_and_return_conditional_losses_35390?
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
 __inference__wrapped_model_35141? !"#$%&'()*+,-./01238?5
.?+
)?&
input_1?????????@@
? ";?8
6
output_1*?'
output_1?????????@@?
D__inference_conv2d_18_layer_call_and_return_conditional_losses_35875l7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????
? ?
)__inference_conv2d_18_layer_call_fn_35884_7?4
-?*
(?%
inputs?????????@@
? " ???????????
D__inference_conv2d_19_layer_call_and_return_conditional_losses_35895l !7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
)__inference_conv2d_19_layer_call_fn_35904_ !7?4
-?*
(?%
inputs?????????
? " ?????????? ?
D__inference_conv2d_20_layer_call_and_return_conditional_losses_35915l"#7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
)__inference_conv2d_20_layer_call_fn_35924_"#7?4
-?*
(?%
inputs????????? 
? " ??????????@?
D__inference_conv2d_21_layer_call_and_return_conditional_losses_35935m$%7?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0??????????
? ?
)__inference_conv2d_21_layer_call_fn_35944`$%7?4
-?*
(?%
inputs?????????@
? "!????????????
D__inference_conv2d_22_layer_call_and_return_conditional_losses_35955m*+7?4
-?*
(?%
inputs?????????
? ".?+
$?!
0??????????
? ?
)__inference_conv2d_22_layer_call_fn_35964`*+7?4
-?*
(?%
inputs?????????
? "!????????????
D__inference_conv2d_23_layer_call_and_return_conditional_losses_35975?,-J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
)__inference_conv2d_23_layer_call_fn_35984?,-J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
D__inference_conv2d_24_layer_call_and_return_conditional_losses_35995?./J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
)__inference_conv2d_24_layer_call_fn_36004?./J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+???????????????????????????@?
D__inference_conv2d_25_layer_call_and_return_conditional_losses_36015?01I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
)__inference_conv2d_25_layer_call_fn_36024?01I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
D__inference_conv2d_26_layer_call_and_return_conditional_losses_35855?23I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
)__inference_conv2d_26_layer_call_fn_35864?23I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
P__inference_convolution_decoder_2_layer_call_and_return_conditional_losses_35587?()*+,-./01230?-
&?#
!?
input_1????????? 
? "??<
5?2
0+???????????????????????????
? ?
5__inference_convolution_decoder_2_layer_call_fn_35617t()*+,-./01230?-
&?#
!?
input_1????????? 
? "2?/+????????????????????????????
P__inference_convolution_encoder_2_layer_call_and_return_conditional_losses_35294m
 !"#$%&'8?5
.?+
)?&
input_1?????????@@
? "%?"
?
0????????? 
? ?
5__inference_convolution_encoder_2_layer_call_fn_35320`
 !"#$%&'8?5
.?+
)?&
input_1?????????@@
? "?????????? ?
V__inference_convolutional_autoencoder_2_layer_call_and_return_conditional_losses_35668? !"#$%&'()*+,-./01238?5
.?+
)?&
input_1?????????@@
? "??<
5?2
0+???????????????????????????
? ?
;__inference_convolutional_autoencoder_2_layer_call_fn_35718? !"#$%&'()*+,-./01238?5
.?+
)?&
input_1?????????@@
? "2?/+????????????????????????????
B__inference_dense_4_layer_call_and_return_conditional_losses_35798]&'0?-
&?#
!?
inputs??????????	
? "%?"
?
0????????? 
? {
'__inference_dense_4_layer_call_fn_35807P&'0?-
&?#
!?
inputs??????????	
? "?????????? ?
B__inference_dense_5_layer_call_and_return_conditional_losses_35817]()/?,
%?"
 ?
inputs????????? 
? "&?#
?
0??????????
? {
'__inference_dense_5_layer_call_fn_35826P()/?,
%?"
 ?
inputs????????? 
? "????????????
D__inference_flatten_2_layer_call_and_return_conditional_losses_35783b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????	
? ?
)__inference_flatten_2_layer_call_fn_35788U8?5
.?+
)?&
inputs??????????
? "???????????	?
D__inference_reshape_2_layer_call_and_return_conditional_losses_35840a0?-
&?#
!?
inputs??????????
? "-?*
#? 
0?????????
? ?
)__inference_reshape_2_layer_call_fn_35845T0?-
&?#
!?
inputs??????????
? " ???????????
#__inference_signature_wrapper_35777? !"#$%&'()*+,-./0123C?@
? 
9?6
4
input_1)?&
input_1?????????@@";?8
6
output_1*?'
output_1?????????@@?
K__inference_up_sampling2d_10_layer_call_and_return_conditional_losses_35371?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_up_sampling2d_10_layer_call_fn_35377?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_up_sampling2d_11_layer_call_and_return_conditional_losses_35390?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_up_sampling2d_11_layer_call_fn_35396?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_35333?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_up_sampling2d_8_layer_call_fn_35339?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_up_sampling2d_9_layer_call_and_return_conditional_losses_35352?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_up_sampling2d_9_layer_call_fn_35358?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????