??7
??
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
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
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
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.22v2.6.1-9-gc2363d6d0258ē/
z
ConA/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConA/kernel
s
ConA/kernel/Read/ReadVariableOpReadVariableOpConA/kernel*&
_output_shapes
:*
dtype0
j
	ConA/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	ConA/bias
c
ConA/bias/Read/ReadVariableOpReadVariableOp	ConA/bias*
_output_shapes
:*
dtype0
z
ConC/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	$*
shared_nameConC/kernel
s
ConC/kernel/Read/ReadVariableOpReadVariableOpConC/kernel*&
_output_shapes
:	$*
dtype0
j
	ConC/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*
shared_name	ConC/bias
c
ConC/bias/Read/ReadVariableOpReadVariableOp	ConC/bias*
_output_shapes
:$*
dtype0
z
ConE/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameConE/kernel
s
ConE/kernel/Read/ReadVariableOpReadVariableOpConE/kernel*&
_output_shapes
:0*
dtype0
j
	ConE/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_name	ConE/bias
c
ConE/bias/Read/ReadVariableOpReadVariableOp	ConE/bias*
_output_shapes
:0*
dtype0
|
ConA2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConA2/kernel
u
 ConA2/kernel/Read/ReadVariableOpReadVariableOpConA2/kernel*&
_output_shapes
:*
dtype0
l

ConA2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
ConA2/bias
e
ConA2/bias/Read/ReadVariableOpReadVariableOp
ConA2/bias*
_output_shapes
:*
dtype0
|
ConC2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*
shared_nameConC2/kernel
u
 ConC2/kernel/Read/ReadVariableOpReadVariableOpConC2/kernel*&
_output_shapes
:$*
dtype0
l

ConC2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
ConC2/bias
e
ConC2/bias/Read/ReadVariableOpReadVariableOp
ConC2/bias*
_output_shapes
:*
dtype0
|
ConE2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameConE2/kernel
u
 ConE2/kernel/Read/ReadVariableOpReadVariableOpConE2/kernel*&
_output_shapes
:0*
dtype0
l

ConE2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
ConE2/bias
e
ConE2/bias/Read/ReadVariableOpReadVariableOp
ConE2/bias*
_output_shapes
:*
dtype0
|
ConA3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConA3/kernel
u
 ConA3/kernel/Read/ReadVariableOpReadVariableOpConA3/kernel*&
_output_shapes
:*
dtype0
l

ConA3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
ConA3/bias
e
ConA3/bias/Read/ReadVariableOpReadVariableOp
ConA3/bias*
_output_shapes
:*
dtype0
|
ConC3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConC3/kernel
u
 ConC3/kernel/Read/ReadVariableOpReadVariableOpConC3/kernel*&
_output_shapes
:*
dtype0
l

ConC3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
ConC3/bias
e
ConC3/bias/Read/ReadVariableOpReadVariableOp
ConC3/bias*
_output_shapes
:*
dtype0
|
ConE3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConE3/kernel
u
 ConE3/kernel/Read/ReadVariableOpReadVariableOpConE3/kernel*&
_output_shapes
:*
dtype0
l

ConE3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
ConE3/bias
e
ConE3/bias/Read/ReadVariableOpReadVariableOp
ConE3/bias*
_output_shapes
:*
dtype0
|
ConA4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConA4/kernel
u
 ConA4/kernel/Read/ReadVariableOpReadVariableOpConA4/kernel*&
_output_shapes
:*
dtype0
l

ConA4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
ConA4/bias
e
ConA4/bias/Read/ReadVariableOpReadVariableOp
ConA4/bias*
_output_shapes
:*
dtype0
|
ConC4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConC4/kernel
u
 ConC4/kernel/Read/ReadVariableOpReadVariableOpConC4/kernel*&
_output_shapes
:*
dtype0
l

ConC4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
ConC4/bias
e
ConC4/bias/Read/ReadVariableOpReadVariableOp
ConC4/bias*
_output_shapes
:*
dtype0
|
ConE4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConE4/kernel
u
 ConE4/kernel/Read/ReadVariableOpReadVariableOpConE4/kernel*&
_output_shapes
:*
dtype0
l

ConE4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
ConE4/bias
e
ConE4/bias/Read/ReadVariableOpReadVariableOp
ConE4/bias*
_output_shapes
:*
dtype0
u
HidA/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*
shared_nameHidA/kernel
n
HidA/kernel/Read/ReadVariableOpReadVariableOpHidA/kernel*!
_output_shapes
:???*
dtype0
k
	HidA/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	HidA/bias
d
HidA/bias/Read/ReadVariableOpReadVariableOp	HidA/bias*
_output_shapes	
:?*
dtype0
w
HidA2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*
shared_nameHidA2/kernel
p
 HidA2/kernel/Read/ReadVariableOpReadVariableOpHidA2/kernel*!
_output_shapes
:???*
dtype0
m

HidA2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
HidA2/bias
f
HidA2/bias/Read/ReadVariableOpReadVariableOp
HidA2/bias*
_output_shapes	
:?*
dtype0
v
HidA3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?#?*
shared_nameHidA3/kernel
o
 HidA3/kernel/Read/ReadVariableOpReadVariableOpHidA3/kernel* 
_output_shapes
:
?#?*
dtype0
m

HidA3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
HidA3/bias
f
HidA3/bias/Read/ReadVariableOpReadVariableOp
HidA3/bias*
_output_shapes	
:?*
dtype0
v
HidA4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameHidA4/kernel
o
 HidA4/kernel/Read/ReadVariableOpReadVariableOpHidA4/kernel* 
_output_shapes
:
??*
dtype0
m

HidA4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
HidA4/bias
f
HidA4/bias/Read/ReadVariableOpReadVariableOp
HidA4/bias*
_output_shapes	
:?*
dtype0
u
HidC/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??	?*
shared_nameHidC/kernel
n
HidC/kernel/Read/ReadVariableOpReadVariableOpHidC/kernel*!
_output_shapes
:??	?*
dtype0
k
	HidC/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	HidC/bias
d
HidC/bias/Read/ReadVariableOpReadVariableOp	HidC/bias*
_output_shapes	
:?*
dtype0
w
HidC2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*
shared_nameHidC2/kernel
p
 HidC2/kernel/Read/ReadVariableOpReadVariableOpHidC2/kernel*!
_output_shapes
:???*
dtype0
m

HidC2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
HidC2/bias
f
HidC2/bias/Read/ReadVariableOpReadVariableOp
HidC2/bias*
_output_shapes	
:?*
dtype0
v
HidC3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
? ?*
shared_nameHidC3/kernel
o
 HidC3/kernel/Read/ReadVariableOpReadVariableOpHidC3/kernel* 
_output_shapes
:
? ?*
dtype0
m

HidC3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
HidC3/bias
f
HidC3/bias/Read/ReadVariableOpReadVariableOp
HidC3/bias*
_output_shapes	
:?*
dtype0
v
HidC4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameHidC4/kernel
o
 HidC4/kernel/Read/ReadVariableOpReadVariableOpHidC4/kernel* 
_output_shapes
:
??*
dtype0
m

HidC4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
HidC4/bias
f
HidC4/bias/Read/ReadVariableOpReadVariableOp
HidC4/bias*
_output_shapes	
:?*
dtype0
u
HidE/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*
shared_nameHidE/kernel
n
HidE/kernel/Read/ReadVariableOpReadVariableOpHidE/kernel*!
_output_shapes
:???*
dtype0
k
	HidE/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	HidE/bias
d
HidE/bias/Read/ReadVariableOpReadVariableOp	HidE/bias*
_output_shapes	
:?*
dtype0
w
HidE2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*
shared_nameHidE2/kernel
p
 HidE2/kernel/Read/ReadVariableOpReadVariableOpHidE2/kernel*!
_output_shapes
:???*
dtype0
m

HidE2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
HidE2/bias
f
HidE2/bias/Read/ReadVariableOpReadVariableOp
HidE2/bias*
_output_shapes	
:?*
dtype0
v
HidE3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameHidE3/kernel
o
 HidE3/kernel/Read/ReadVariableOpReadVariableOpHidE3/kernel* 
_output_shapes
:
??*
dtype0
m

HidE3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
HidE3/bias
f
HidE3/bias/Read/ReadVariableOpReadVariableOp
HidE3/bias*
_output_shapes	
:?*
dtype0
v
HidE4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameHidE4/kernel
o
 HidE4/kernel/Read/ReadVariableOpReadVariableOpHidE4/kernel* 
_output_shapes
:
??*
dtype0
m

HidE4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
HidE4/bias
f
HidE4/bias/Read/ReadVariableOpReadVariableOp
HidE4/bias*
_output_shapes	
:?*
dtype0
t
Hid1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameHid1/kernel
m
Hid1/kernel/Read/ReadVariableOpReadVariableOpHid1/kernel* 
_output_shapes
:
??*
dtype0
k
	Hid1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	Hid1/bias
d
Hid1/bias/Read/ReadVariableOpReadVariableOp	Hid1/bias*
_output_shapes	
:?*
dtype0
t
Hid2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameHid2/kernel
m
Hid2/kernel/Read/ReadVariableOpReadVariableOpHid2/kernel* 
_output_shapes
:
??*
dtype0
k
	Hid2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	Hid2/bias
d
Hid2/bias/Read/ReadVariableOpReadVariableOp	Hid2/bias*
_output_shapes	
:?*
dtype0
w
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?(*
shared_nameoutput/kernel
p
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes
:	?(*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:(*
dtype0
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/ConA/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/ConA/kernel/m
?
&Adam/ConA/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConA/kernel/m*&
_output_shapes
:*
dtype0
x
Adam/ConA/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/ConA/bias/m
q
$Adam/ConA/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConA/bias/m*
_output_shapes
:*
dtype0
?
Adam/ConC/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	$*#
shared_nameAdam/ConC/kernel/m
?
&Adam/ConC/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConC/kernel/m*&
_output_shapes
:	$*
dtype0
x
Adam/ConC/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*!
shared_nameAdam/ConC/bias/m
q
$Adam/ConC/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConC/bias/m*
_output_shapes
:$*
dtype0
?
Adam/ConE/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*#
shared_nameAdam/ConE/kernel/m
?
&Adam/ConE/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConE/kernel/m*&
_output_shapes
:0*
dtype0
x
Adam/ConE/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*!
shared_nameAdam/ConE/bias/m
q
$Adam/ConE/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConE/bias/m*
_output_shapes
:0*
dtype0
?
Adam/ConA2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConA2/kernel/m
?
'Adam/ConA2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConA2/kernel/m*&
_output_shapes
:*
dtype0
z
Adam/ConA2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/ConA2/bias/m
s
%Adam/ConA2/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConA2/bias/m*
_output_shapes
:*
dtype0
?
Adam/ConC2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*$
shared_nameAdam/ConC2/kernel/m
?
'Adam/ConC2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConC2/kernel/m*&
_output_shapes
:$*
dtype0
z
Adam/ConC2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/ConC2/bias/m
s
%Adam/ConC2/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConC2/bias/m*
_output_shapes
:*
dtype0
?
Adam/ConE2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*$
shared_nameAdam/ConE2/kernel/m
?
'Adam/ConE2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConE2/kernel/m*&
_output_shapes
:0*
dtype0
z
Adam/ConE2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/ConE2/bias/m
s
%Adam/ConE2/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConE2/bias/m*
_output_shapes
:*
dtype0
?
Adam/ConA3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConA3/kernel/m
?
'Adam/ConA3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConA3/kernel/m*&
_output_shapes
:*
dtype0
z
Adam/ConA3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/ConA3/bias/m
s
%Adam/ConA3/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConA3/bias/m*
_output_shapes
:*
dtype0
?
Adam/ConC3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConC3/kernel/m
?
'Adam/ConC3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConC3/kernel/m*&
_output_shapes
:*
dtype0
z
Adam/ConC3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/ConC3/bias/m
s
%Adam/ConC3/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConC3/bias/m*
_output_shapes
:*
dtype0
?
Adam/ConE3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConE3/kernel/m
?
'Adam/ConE3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConE3/kernel/m*&
_output_shapes
:*
dtype0
z
Adam/ConE3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/ConE3/bias/m
s
%Adam/ConE3/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConE3/bias/m*
_output_shapes
:*
dtype0
?
Adam/ConA4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConA4/kernel/m
?
'Adam/ConA4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConA4/kernel/m*&
_output_shapes
:*
dtype0
z
Adam/ConA4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/ConA4/bias/m
s
%Adam/ConA4/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConA4/bias/m*
_output_shapes
:*
dtype0
?
Adam/ConC4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConC4/kernel/m
?
'Adam/ConC4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConC4/kernel/m*&
_output_shapes
:*
dtype0
z
Adam/ConC4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/ConC4/bias/m
s
%Adam/ConC4/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConC4/bias/m*
_output_shapes
:*
dtype0
?
Adam/ConE4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConE4/kernel/m
?
'Adam/ConE4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConE4/kernel/m*&
_output_shapes
:*
dtype0
z
Adam/ConE4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/ConE4/bias/m
s
%Adam/ConE4/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConE4/bias/m*
_output_shapes
:*
dtype0
?
Adam/HidA/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*#
shared_nameAdam/HidA/kernel/m
|
&Adam/HidA/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidA/kernel/m*!
_output_shapes
:???*
dtype0
y
Adam/HidA/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/HidA/bias/m
r
$Adam/HidA/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidA/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/HidA2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*$
shared_nameAdam/HidA2/kernel/m
~
'Adam/HidA2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidA2/kernel/m*!
_output_shapes
:???*
dtype0
{
Adam/HidA2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidA2/bias/m
t
%Adam/HidA2/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidA2/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/HidA3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?#?*$
shared_nameAdam/HidA3/kernel/m
}
'Adam/HidA3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidA3/kernel/m* 
_output_shapes
:
?#?*
dtype0
{
Adam/HidA3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidA3/bias/m
t
%Adam/HidA3/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidA3/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/HidA4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/HidA4/kernel/m
}
'Adam/HidA4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidA4/kernel/m* 
_output_shapes
:
??*
dtype0
{
Adam/HidA4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidA4/bias/m
t
%Adam/HidA4/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidA4/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/HidC/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??	?*#
shared_nameAdam/HidC/kernel/m
|
&Adam/HidC/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidC/kernel/m*!
_output_shapes
:??	?*
dtype0
y
Adam/HidC/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/HidC/bias/m
r
$Adam/HidC/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidC/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/HidC2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*$
shared_nameAdam/HidC2/kernel/m
~
'Adam/HidC2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidC2/kernel/m*!
_output_shapes
:???*
dtype0
{
Adam/HidC2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidC2/bias/m
t
%Adam/HidC2/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidC2/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/HidC3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
? ?*$
shared_nameAdam/HidC3/kernel/m
}
'Adam/HidC3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidC3/kernel/m* 
_output_shapes
:
? ?*
dtype0
{
Adam/HidC3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidC3/bias/m
t
%Adam/HidC3/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidC3/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/HidC4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/HidC4/kernel/m
}
'Adam/HidC4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidC4/kernel/m* 
_output_shapes
:
??*
dtype0
{
Adam/HidC4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidC4/bias/m
t
%Adam/HidC4/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidC4/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/HidE/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*#
shared_nameAdam/HidE/kernel/m
|
&Adam/HidE/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidE/kernel/m*!
_output_shapes
:???*
dtype0
y
Adam/HidE/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/HidE/bias/m
r
$Adam/HidE/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidE/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/HidE2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*$
shared_nameAdam/HidE2/kernel/m
~
'Adam/HidE2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidE2/kernel/m*!
_output_shapes
:???*
dtype0
{
Adam/HidE2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidE2/bias/m
t
%Adam/HidE2/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidE2/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/HidE3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/HidE3/kernel/m
}
'Adam/HidE3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidE3/kernel/m* 
_output_shapes
:
??*
dtype0
{
Adam/HidE3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidE3/bias/m
t
%Adam/HidE3/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidE3/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/HidE4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/HidE4/kernel/m
}
'Adam/HidE4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidE4/kernel/m* 
_output_shapes
:
??*
dtype0
{
Adam/HidE4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidE4/bias/m
t
%Adam/HidE4/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidE4/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/Hid1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_nameAdam/Hid1/kernel/m
{
&Adam/Hid1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Hid1/kernel/m* 
_output_shapes
:
??*
dtype0
y
Adam/Hid1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/Hid1/bias/m
r
$Adam/Hid1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Hid1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/Hid2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_nameAdam/Hid2/kernel/m
{
&Adam/Hid2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Hid2/kernel/m* 
_output_shapes
:
??*
dtype0
y
Adam/Hid2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/Hid2/bias/m
r
$Adam/Hid2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Hid2/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?(*%
shared_nameAdam/output/kernel/m
~
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*
_output_shapes
:	?(*
dtype0
|
Adam/output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*#
shared_nameAdam/output/bias/m
u
&Adam/output/bias/m/Read/ReadVariableOpReadVariableOpAdam/output/bias/m*
_output_shapes
:(*
dtype0
?
Adam/ConA/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/ConA/kernel/v
?
&Adam/ConA/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConA/kernel/v*&
_output_shapes
:*
dtype0
x
Adam/ConA/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/ConA/bias/v
q
$Adam/ConA/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConA/bias/v*
_output_shapes
:*
dtype0
?
Adam/ConC/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	$*#
shared_nameAdam/ConC/kernel/v
?
&Adam/ConC/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConC/kernel/v*&
_output_shapes
:	$*
dtype0
x
Adam/ConC/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*!
shared_nameAdam/ConC/bias/v
q
$Adam/ConC/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConC/bias/v*
_output_shapes
:$*
dtype0
?
Adam/ConE/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*#
shared_nameAdam/ConE/kernel/v
?
&Adam/ConE/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConE/kernel/v*&
_output_shapes
:0*
dtype0
x
Adam/ConE/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*!
shared_nameAdam/ConE/bias/v
q
$Adam/ConE/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConE/bias/v*
_output_shapes
:0*
dtype0
?
Adam/ConA2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConA2/kernel/v
?
'Adam/ConA2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConA2/kernel/v*&
_output_shapes
:*
dtype0
z
Adam/ConA2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/ConA2/bias/v
s
%Adam/ConA2/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConA2/bias/v*
_output_shapes
:*
dtype0
?
Adam/ConC2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*$
shared_nameAdam/ConC2/kernel/v
?
'Adam/ConC2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConC2/kernel/v*&
_output_shapes
:$*
dtype0
z
Adam/ConC2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/ConC2/bias/v
s
%Adam/ConC2/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConC2/bias/v*
_output_shapes
:*
dtype0
?
Adam/ConE2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*$
shared_nameAdam/ConE2/kernel/v
?
'Adam/ConE2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConE2/kernel/v*&
_output_shapes
:0*
dtype0
z
Adam/ConE2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/ConE2/bias/v
s
%Adam/ConE2/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConE2/bias/v*
_output_shapes
:*
dtype0
?
Adam/ConA3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConA3/kernel/v
?
'Adam/ConA3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConA3/kernel/v*&
_output_shapes
:*
dtype0
z
Adam/ConA3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/ConA3/bias/v
s
%Adam/ConA3/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConA3/bias/v*
_output_shapes
:*
dtype0
?
Adam/ConC3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConC3/kernel/v
?
'Adam/ConC3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConC3/kernel/v*&
_output_shapes
:*
dtype0
z
Adam/ConC3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/ConC3/bias/v
s
%Adam/ConC3/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConC3/bias/v*
_output_shapes
:*
dtype0
?
Adam/ConE3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConE3/kernel/v
?
'Adam/ConE3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConE3/kernel/v*&
_output_shapes
:*
dtype0
z
Adam/ConE3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/ConE3/bias/v
s
%Adam/ConE3/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConE3/bias/v*
_output_shapes
:*
dtype0
?
Adam/ConA4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConA4/kernel/v
?
'Adam/ConA4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConA4/kernel/v*&
_output_shapes
:*
dtype0
z
Adam/ConA4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/ConA4/bias/v
s
%Adam/ConA4/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConA4/bias/v*
_output_shapes
:*
dtype0
?
Adam/ConC4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConC4/kernel/v
?
'Adam/ConC4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConC4/kernel/v*&
_output_shapes
:*
dtype0
z
Adam/ConC4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/ConC4/bias/v
s
%Adam/ConC4/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConC4/bias/v*
_output_shapes
:*
dtype0
?
Adam/ConE4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConE4/kernel/v
?
'Adam/ConE4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConE4/kernel/v*&
_output_shapes
:*
dtype0
z
Adam/ConE4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/ConE4/bias/v
s
%Adam/ConE4/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConE4/bias/v*
_output_shapes
:*
dtype0
?
Adam/HidA/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*#
shared_nameAdam/HidA/kernel/v
|
&Adam/HidA/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidA/kernel/v*!
_output_shapes
:???*
dtype0
y
Adam/HidA/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/HidA/bias/v
r
$Adam/HidA/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidA/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/HidA2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*$
shared_nameAdam/HidA2/kernel/v
~
'Adam/HidA2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidA2/kernel/v*!
_output_shapes
:???*
dtype0
{
Adam/HidA2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidA2/bias/v
t
%Adam/HidA2/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidA2/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/HidA3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?#?*$
shared_nameAdam/HidA3/kernel/v
}
'Adam/HidA3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidA3/kernel/v* 
_output_shapes
:
?#?*
dtype0
{
Adam/HidA3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidA3/bias/v
t
%Adam/HidA3/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidA3/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/HidA4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/HidA4/kernel/v
}
'Adam/HidA4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidA4/kernel/v* 
_output_shapes
:
??*
dtype0
{
Adam/HidA4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidA4/bias/v
t
%Adam/HidA4/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidA4/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/HidC/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??	?*#
shared_nameAdam/HidC/kernel/v
|
&Adam/HidC/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidC/kernel/v*!
_output_shapes
:??	?*
dtype0
y
Adam/HidC/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/HidC/bias/v
r
$Adam/HidC/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidC/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/HidC2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*$
shared_nameAdam/HidC2/kernel/v
~
'Adam/HidC2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidC2/kernel/v*!
_output_shapes
:???*
dtype0
{
Adam/HidC2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidC2/bias/v
t
%Adam/HidC2/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidC2/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/HidC3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
? ?*$
shared_nameAdam/HidC3/kernel/v
}
'Adam/HidC3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidC3/kernel/v* 
_output_shapes
:
? ?*
dtype0
{
Adam/HidC3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidC3/bias/v
t
%Adam/HidC3/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidC3/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/HidC4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/HidC4/kernel/v
}
'Adam/HidC4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidC4/kernel/v* 
_output_shapes
:
??*
dtype0
{
Adam/HidC4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidC4/bias/v
t
%Adam/HidC4/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidC4/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/HidE/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*#
shared_nameAdam/HidE/kernel/v
|
&Adam/HidE/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidE/kernel/v*!
_output_shapes
:???*
dtype0
y
Adam/HidE/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/HidE/bias/v
r
$Adam/HidE/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidE/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/HidE2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*$
shared_nameAdam/HidE2/kernel/v
~
'Adam/HidE2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidE2/kernel/v*!
_output_shapes
:???*
dtype0
{
Adam/HidE2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidE2/bias/v
t
%Adam/HidE2/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidE2/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/HidE3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/HidE3/kernel/v
}
'Adam/HidE3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidE3/kernel/v* 
_output_shapes
:
??*
dtype0
{
Adam/HidE3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidE3/bias/v
t
%Adam/HidE3/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidE3/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/HidE4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/HidE4/kernel/v
}
'Adam/HidE4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidE4/kernel/v* 
_output_shapes
:
??*
dtype0
{
Adam/HidE4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidE4/bias/v
t
%Adam/HidE4/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidE4/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/Hid1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_nameAdam/Hid1/kernel/v
{
&Adam/Hid1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Hid1/kernel/v* 
_output_shapes
:
??*
dtype0
y
Adam/Hid1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/Hid1/bias/v
r
$Adam/Hid1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Hid1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/Hid2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_nameAdam/Hid2/kernel/v
{
&Adam/Hid2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Hid2/kernel/v* 
_output_shapes
:
??*
dtype0
y
Adam/Hid2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/Hid2/bias/v
r
$Adam/Hid2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Hid2/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?(*%
shared_nameAdam/output/kernel/v
~
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*
_output_shapes
:	?(*
dtype0
|
Adam/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*#
shared_nameAdam/output/bias/v
u
&Adam/output/bias/v/Read/ReadVariableOpReadVariableOpAdam/output/bias/v*
_output_shapes
:(*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
layer_with_weights-7
layer-17
layer_with_weights-8
layer-18
layer-19
layer-20
layer-21
layer_with_weights-9
layer-22
layer_with_weights-10
layer-23
layer_with_weights-11
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
layer-30
 layer-31
!layer-32
"layer-33
#layer-34
$layer-35
%layer-36
&layer-37
'layer-38
(layer-39
)layer-40
*layer-41
+layer-42
,layer-43
-layer-44
.layer-45
/layer-46
0layer-47
1layer-48
2layer_with_weights-12
2layer-49
3layer_with_weights-13
3layer-50
4layer_with_weights-14
4layer-51
5layer_with_weights-15
5layer-52
6layer_with_weights-16
6layer-53
7layer_with_weights-17
7layer-54
8layer_with_weights-18
8layer-55
9layer_with_weights-19
9layer-56
:layer_with_weights-20
:layer-57
;layer_with_weights-21
;layer-58
<layer_with_weights-22
<layer-59
=layer_with_weights-23
=layer-60
>layer-61
?layer_with_weights-24
?layer-62
@layer_with_weights-25
@layer-63
Alayer_with_weights-26
Alayer-64
B	optimizer
C
signatures
#D_self_saveable_object_factories
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
%
#I_self_saveable_object_factories
?

Jkernel
Kbias
#L_self_saveable_object_factories
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
?

Qkernel
Rbias
#S_self_saveable_object_factories
Ttrainable_variables
U	variables
Vregularization_losses
W	keras_api
?

Xkernel
Ybias
#Z_self_saveable_object_factories
[trainable_variables
\	variables
]regularization_losses
^	keras_api
w
#__self_saveable_object_factories
`trainable_variables
a	variables
bregularization_losses
c	keras_api
w
#d_self_saveable_object_factories
etrainable_variables
f	variables
gregularization_losses
h	keras_api
w
#i_self_saveable_object_factories
jtrainable_variables
k	variables
lregularization_losses
m	keras_api
w
#n_self_saveable_object_factories
otrainable_variables
p	variables
qregularization_losses
r	keras_api
w
#s_self_saveable_object_factories
ttrainable_variables
u	variables
vregularization_losses
w	keras_api
w
#x_self_saveable_object_factories
ytrainable_variables
z	variables
{regularization_losses
|	keras_api
?

}kernel
~bias
#_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?	
	?iter
?beta_1
?beta_2

?decay
?learning_rateJm?Km?Qm?Rm?Xm?Ym?}m?~m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?Jv?Kv?Qv?Rv?Xv?Yv?}v?~v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?
 
 
?
J0
K1
Q2
R3
X4
Y5
}6
~7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53
?
J0
K1
Q2
R3
X4
Y5
}6
~7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53
 
?
?non_trainable_variables
?layer_metrics
Etrainable_variables
?metrics
F	variables
 ?layer_regularization_losses
Gregularization_losses
?layers
 
WU
VARIABLE_VALUEConA/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	ConA/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

J0
K1

J0
K1
 
?
?non_trainable_variables
?layer_metrics
Mtrainable_variables
?metrics
N	variables
 ?layer_regularization_losses
Oregularization_losses
?layers
WU
VARIABLE_VALUEConC/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	ConC/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Q0
R1

Q0
R1
 
?
?non_trainable_variables
?layer_metrics
Ttrainable_variables
?metrics
U	variables
 ?layer_regularization_losses
Vregularization_losses
?layers
WU
VARIABLE_VALUEConE/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	ConE/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

X0
Y1

X0
Y1
 
?
?non_trainable_variables
?layer_metrics
[trainable_variables
?metrics
\	variables
 ?layer_regularization_losses
]regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
`trainable_variables
?metrics
a	variables
 ?layer_regularization_losses
bregularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
etrainable_variables
?metrics
f	variables
 ?layer_regularization_losses
gregularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
jtrainable_variables
?metrics
k	variables
 ?layer_regularization_losses
lregularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
otrainable_variables
?metrics
p	variables
 ?layer_regularization_losses
qregularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
ttrainable_variables
?metrics
u	variables
 ?layer_regularization_losses
vregularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
ytrainable_variables
?metrics
z	variables
 ?layer_regularization_losses
{regularization_losses
?layers
XV
VARIABLE_VALUEConA2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
ConA2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

}0
~1

}0
~1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
XV
VARIABLE_VALUEConC2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
ConC2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
XV
VARIABLE_VALUEConE2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
ConE2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
XV
VARIABLE_VALUEConA3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
ConA3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
XV
VARIABLE_VALUEConC3/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
ConC3/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
XV
VARIABLE_VALUEConE3/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
ConE3/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
XV
VARIABLE_VALUEConA4/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
ConA4/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
YW
VARIABLE_VALUEConC4/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
ConC4/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
YW
VARIABLE_VALUEConE4/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
ConE4/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
XV
VARIABLE_VALUEHidA/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	HidA/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
YW
VARIABLE_VALUEHidA2/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidA2/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
YW
VARIABLE_VALUEHidA3/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidA3/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
YW
VARIABLE_VALUEHidA4/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidA4/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
XV
VARIABLE_VALUEHidC/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	HidC/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
YW
VARIABLE_VALUEHidC2/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidC2/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
YW
VARIABLE_VALUEHidC3/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidC3/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
YW
VARIABLE_VALUEHidC4/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidC4/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
XV
VARIABLE_VALUEHidE/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	HidE/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
YW
VARIABLE_VALUEHidE2/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidE2/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
YW
VARIABLE_VALUEHidE3/kernel7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidE3/bias5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
YW
VARIABLE_VALUEHidE4/kernel7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidE4/bias5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
XV
VARIABLE_VALUEHid1/kernel7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	Hid1/bias5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
XV
VARIABLE_VALUEHid2/kernel7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	Hid2/bias5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
ZX
VARIABLE_VALUEoutput/kernel7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEoutput/bias5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
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
 
 

?0
?1
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653
754
855
956
:57
;58
<59
=60
>61
?62
@63
A64
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
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
zx
VARIABLE_VALUEAdam/ConA/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConA/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConC/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConC/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConE/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConE/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConA2/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConA2/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConC2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConC2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConE2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConE2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConA3/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConA3/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConC3/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConC3/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConE3/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConE3/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConA4/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConA4/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/ConC4/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/ConC4/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/ConE4/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/ConE4/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/HidA/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/HidA/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidA2/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidA2/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidA3/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidA3/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidA4/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidA4/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/HidC/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/HidC/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidC2/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidC2/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidC3/kernel/mSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidC3/bias/mQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidC4/kernel/mSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidC4/bias/mQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/HidE/kernel/mSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/HidE/bias/mQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidE2/kernel/mSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidE2/bias/mQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidE3/kernel/mSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidE3/bias/mQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidE4/kernel/mSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidE4/bias/mQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/Hid1/kernel/mSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Hid1/bias/mQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/Hid2/kernel/mSlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Hid2/bias/mQlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/output/kernel/mSlayer_with_weights-26/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/output/bias/mQlayer_with_weights-26/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConA/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConA/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConC/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConC/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConE/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConE/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConA2/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConA2/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConC2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConC2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConE2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConE2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConA3/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConA3/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConC3/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConC3/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConE3/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConE3/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConA4/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConA4/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/ConC4/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/ConC4/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/ConE4/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/ConE4/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/HidA/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/HidA/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidA2/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidA2/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidA3/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidA3/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidA4/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidA4/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/HidC/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/HidC/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidC2/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidC2/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidC3/kernel/vSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidC3/bias/vQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidC4/kernel/vSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidC4/bias/vQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/HidE/kernel/vSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/HidE/bias/vQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidE2/kernel/vSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidE2/bias/vQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidE3/kernel/vSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidE3/bias/vQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidE4/kernel/vSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidE4/bias/vQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/Hid1/kernel/vSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Hid1/bias/vQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/Hid2/kernel/vSlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Hid2/bias/vQlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/output/kernel/vSlayer_with_weights-26/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/output/bias/vQlayer_with_weights-26/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_inputPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputConE/kernel	ConE/biasConC/kernel	ConC/biasConA/kernel	ConA/biasConE2/kernel
ConE2/biasConC2/kernel
ConC2/biasConA2/kernel
ConA2/biasConE3/kernel
ConE3/biasConC3/kernel
ConC3/biasConA3/kernel
ConA3/biasConE4/kernel
ConE4/biasConC4/kernel
ConC4/biasConA4/kernel
ConA4/biasHidA/kernel	HidA/biasHidA2/kernel
HidA2/biasHidA3/kernel
HidA3/biasHidA4/kernel
HidA4/biasHidC/kernel	HidC/biasHidC2/kernel
HidC2/biasHidC3/kernel
HidC3/biasHidC4/kernel
HidC4/biasHidE/kernel	HidE/biasHidE2/kernel
HidE2/biasHidE3/kernel
HidE3/biasHidE4/kernel
HidE4/biasHid1/kernel	Hid1/biasHid2/kernel	Hid2/biasoutput/kerneloutput/bias*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_229127
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?5
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameConA/kernel/Read/ReadVariableOpConA/bias/Read/ReadVariableOpConC/kernel/Read/ReadVariableOpConC/bias/Read/ReadVariableOpConE/kernel/Read/ReadVariableOpConE/bias/Read/ReadVariableOp ConA2/kernel/Read/ReadVariableOpConA2/bias/Read/ReadVariableOp ConC2/kernel/Read/ReadVariableOpConC2/bias/Read/ReadVariableOp ConE2/kernel/Read/ReadVariableOpConE2/bias/Read/ReadVariableOp ConA3/kernel/Read/ReadVariableOpConA3/bias/Read/ReadVariableOp ConC3/kernel/Read/ReadVariableOpConC3/bias/Read/ReadVariableOp ConE3/kernel/Read/ReadVariableOpConE3/bias/Read/ReadVariableOp ConA4/kernel/Read/ReadVariableOpConA4/bias/Read/ReadVariableOp ConC4/kernel/Read/ReadVariableOpConC4/bias/Read/ReadVariableOp ConE4/kernel/Read/ReadVariableOpConE4/bias/Read/ReadVariableOpHidA/kernel/Read/ReadVariableOpHidA/bias/Read/ReadVariableOp HidA2/kernel/Read/ReadVariableOpHidA2/bias/Read/ReadVariableOp HidA3/kernel/Read/ReadVariableOpHidA3/bias/Read/ReadVariableOp HidA4/kernel/Read/ReadVariableOpHidA4/bias/Read/ReadVariableOpHidC/kernel/Read/ReadVariableOpHidC/bias/Read/ReadVariableOp HidC2/kernel/Read/ReadVariableOpHidC2/bias/Read/ReadVariableOp HidC3/kernel/Read/ReadVariableOpHidC3/bias/Read/ReadVariableOp HidC4/kernel/Read/ReadVariableOpHidC4/bias/Read/ReadVariableOpHidE/kernel/Read/ReadVariableOpHidE/bias/Read/ReadVariableOp HidE2/kernel/Read/ReadVariableOpHidE2/bias/Read/ReadVariableOp HidE3/kernel/Read/ReadVariableOpHidE3/bias/Read/ReadVariableOp HidE4/kernel/Read/ReadVariableOpHidE4/bias/Read/ReadVariableOpHid1/kernel/Read/ReadVariableOpHid1/bias/Read/ReadVariableOpHid2/kernel/Read/ReadVariableOpHid2/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp&Adam/ConA/kernel/m/Read/ReadVariableOp$Adam/ConA/bias/m/Read/ReadVariableOp&Adam/ConC/kernel/m/Read/ReadVariableOp$Adam/ConC/bias/m/Read/ReadVariableOp&Adam/ConE/kernel/m/Read/ReadVariableOp$Adam/ConE/bias/m/Read/ReadVariableOp'Adam/ConA2/kernel/m/Read/ReadVariableOp%Adam/ConA2/bias/m/Read/ReadVariableOp'Adam/ConC2/kernel/m/Read/ReadVariableOp%Adam/ConC2/bias/m/Read/ReadVariableOp'Adam/ConE2/kernel/m/Read/ReadVariableOp%Adam/ConE2/bias/m/Read/ReadVariableOp'Adam/ConA3/kernel/m/Read/ReadVariableOp%Adam/ConA3/bias/m/Read/ReadVariableOp'Adam/ConC3/kernel/m/Read/ReadVariableOp%Adam/ConC3/bias/m/Read/ReadVariableOp'Adam/ConE3/kernel/m/Read/ReadVariableOp%Adam/ConE3/bias/m/Read/ReadVariableOp'Adam/ConA4/kernel/m/Read/ReadVariableOp%Adam/ConA4/bias/m/Read/ReadVariableOp'Adam/ConC4/kernel/m/Read/ReadVariableOp%Adam/ConC4/bias/m/Read/ReadVariableOp'Adam/ConE4/kernel/m/Read/ReadVariableOp%Adam/ConE4/bias/m/Read/ReadVariableOp&Adam/HidA/kernel/m/Read/ReadVariableOp$Adam/HidA/bias/m/Read/ReadVariableOp'Adam/HidA2/kernel/m/Read/ReadVariableOp%Adam/HidA2/bias/m/Read/ReadVariableOp'Adam/HidA3/kernel/m/Read/ReadVariableOp%Adam/HidA3/bias/m/Read/ReadVariableOp'Adam/HidA4/kernel/m/Read/ReadVariableOp%Adam/HidA4/bias/m/Read/ReadVariableOp&Adam/HidC/kernel/m/Read/ReadVariableOp$Adam/HidC/bias/m/Read/ReadVariableOp'Adam/HidC2/kernel/m/Read/ReadVariableOp%Adam/HidC2/bias/m/Read/ReadVariableOp'Adam/HidC3/kernel/m/Read/ReadVariableOp%Adam/HidC3/bias/m/Read/ReadVariableOp'Adam/HidC4/kernel/m/Read/ReadVariableOp%Adam/HidC4/bias/m/Read/ReadVariableOp&Adam/HidE/kernel/m/Read/ReadVariableOp$Adam/HidE/bias/m/Read/ReadVariableOp'Adam/HidE2/kernel/m/Read/ReadVariableOp%Adam/HidE2/bias/m/Read/ReadVariableOp'Adam/HidE3/kernel/m/Read/ReadVariableOp%Adam/HidE3/bias/m/Read/ReadVariableOp'Adam/HidE4/kernel/m/Read/ReadVariableOp%Adam/HidE4/bias/m/Read/ReadVariableOp&Adam/Hid1/kernel/m/Read/ReadVariableOp$Adam/Hid1/bias/m/Read/ReadVariableOp&Adam/Hid2/kernel/m/Read/ReadVariableOp$Adam/Hid2/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp&Adam/ConA/kernel/v/Read/ReadVariableOp$Adam/ConA/bias/v/Read/ReadVariableOp&Adam/ConC/kernel/v/Read/ReadVariableOp$Adam/ConC/bias/v/Read/ReadVariableOp&Adam/ConE/kernel/v/Read/ReadVariableOp$Adam/ConE/bias/v/Read/ReadVariableOp'Adam/ConA2/kernel/v/Read/ReadVariableOp%Adam/ConA2/bias/v/Read/ReadVariableOp'Adam/ConC2/kernel/v/Read/ReadVariableOp%Adam/ConC2/bias/v/Read/ReadVariableOp'Adam/ConE2/kernel/v/Read/ReadVariableOp%Adam/ConE2/bias/v/Read/ReadVariableOp'Adam/ConA3/kernel/v/Read/ReadVariableOp%Adam/ConA3/bias/v/Read/ReadVariableOp'Adam/ConC3/kernel/v/Read/ReadVariableOp%Adam/ConC3/bias/v/Read/ReadVariableOp'Adam/ConE3/kernel/v/Read/ReadVariableOp%Adam/ConE3/bias/v/Read/ReadVariableOp'Adam/ConA4/kernel/v/Read/ReadVariableOp%Adam/ConA4/bias/v/Read/ReadVariableOp'Adam/ConC4/kernel/v/Read/ReadVariableOp%Adam/ConC4/bias/v/Read/ReadVariableOp'Adam/ConE4/kernel/v/Read/ReadVariableOp%Adam/ConE4/bias/v/Read/ReadVariableOp&Adam/HidA/kernel/v/Read/ReadVariableOp$Adam/HidA/bias/v/Read/ReadVariableOp'Adam/HidA2/kernel/v/Read/ReadVariableOp%Adam/HidA2/bias/v/Read/ReadVariableOp'Adam/HidA3/kernel/v/Read/ReadVariableOp%Adam/HidA3/bias/v/Read/ReadVariableOp'Adam/HidA4/kernel/v/Read/ReadVariableOp%Adam/HidA4/bias/v/Read/ReadVariableOp&Adam/HidC/kernel/v/Read/ReadVariableOp$Adam/HidC/bias/v/Read/ReadVariableOp'Adam/HidC2/kernel/v/Read/ReadVariableOp%Adam/HidC2/bias/v/Read/ReadVariableOp'Adam/HidC3/kernel/v/Read/ReadVariableOp%Adam/HidC3/bias/v/Read/ReadVariableOp'Adam/HidC4/kernel/v/Read/ReadVariableOp%Adam/HidC4/bias/v/Read/ReadVariableOp&Adam/HidE/kernel/v/Read/ReadVariableOp$Adam/HidE/bias/v/Read/ReadVariableOp'Adam/HidE2/kernel/v/Read/ReadVariableOp%Adam/HidE2/bias/v/Read/ReadVariableOp'Adam/HidE3/kernel/v/Read/ReadVariableOp%Adam/HidE3/bias/v/Read/ReadVariableOp'Adam/HidE4/kernel/v/Read/ReadVariableOp%Adam/HidE4/bias/v/Read/ReadVariableOp&Adam/Hid1/kernel/v/Read/ReadVariableOp$Adam/Hid1/bias/v/Read/ReadVariableOp&Adam/Hid2/kernel/v/Read/ReadVariableOp$Adam/Hid2/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*?
Tin?
?2?	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_231728
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConA/kernel	ConA/biasConC/kernel	ConC/biasConE/kernel	ConE/biasConA2/kernel
ConA2/biasConC2/kernel
ConC2/biasConE2/kernel
ConE2/biasConA3/kernel
ConA3/biasConC3/kernel
ConC3/biasConE3/kernel
ConE3/biasConA4/kernel
ConA4/biasConC4/kernel
ConC4/biasConE4/kernel
ConE4/biasHidA/kernel	HidA/biasHidA2/kernel
HidA2/biasHidA3/kernel
HidA3/biasHidA4/kernel
HidA4/biasHidC/kernel	HidC/biasHidC2/kernel
HidC2/biasHidC3/kernel
HidC3/biasHidC4/kernel
HidC4/biasHidE/kernel	HidE/biasHidE2/kernel
HidE2/biasHidE3/kernel
HidE3/biasHidE4/kernel
HidE4/biasHid1/kernel	Hid1/biasHid2/kernel	Hid2/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/ConA/kernel/mAdam/ConA/bias/mAdam/ConC/kernel/mAdam/ConC/bias/mAdam/ConE/kernel/mAdam/ConE/bias/mAdam/ConA2/kernel/mAdam/ConA2/bias/mAdam/ConC2/kernel/mAdam/ConC2/bias/mAdam/ConE2/kernel/mAdam/ConE2/bias/mAdam/ConA3/kernel/mAdam/ConA3/bias/mAdam/ConC3/kernel/mAdam/ConC3/bias/mAdam/ConE3/kernel/mAdam/ConE3/bias/mAdam/ConA4/kernel/mAdam/ConA4/bias/mAdam/ConC4/kernel/mAdam/ConC4/bias/mAdam/ConE4/kernel/mAdam/ConE4/bias/mAdam/HidA/kernel/mAdam/HidA/bias/mAdam/HidA2/kernel/mAdam/HidA2/bias/mAdam/HidA3/kernel/mAdam/HidA3/bias/mAdam/HidA4/kernel/mAdam/HidA4/bias/mAdam/HidC/kernel/mAdam/HidC/bias/mAdam/HidC2/kernel/mAdam/HidC2/bias/mAdam/HidC3/kernel/mAdam/HidC3/bias/mAdam/HidC4/kernel/mAdam/HidC4/bias/mAdam/HidE/kernel/mAdam/HidE/bias/mAdam/HidE2/kernel/mAdam/HidE2/bias/mAdam/HidE3/kernel/mAdam/HidE3/bias/mAdam/HidE4/kernel/mAdam/HidE4/bias/mAdam/Hid1/kernel/mAdam/Hid1/bias/mAdam/Hid2/kernel/mAdam/Hid2/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/ConA/kernel/vAdam/ConA/bias/vAdam/ConC/kernel/vAdam/ConC/bias/vAdam/ConE/kernel/vAdam/ConE/bias/vAdam/ConA2/kernel/vAdam/ConA2/bias/vAdam/ConC2/kernel/vAdam/ConC2/bias/vAdam/ConE2/kernel/vAdam/ConE2/bias/vAdam/ConA3/kernel/vAdam/ConA3/bias/vAdam/ConC3/kernel/vAdam/ConC3/bias/vAdam/ConE3/kernel/vAdam/ConE3/bias/vAdam/ConA4/kernel/vAdam/ConA4/bias/vAdam/ConC4/kernel/vAdam/ConC4/bias/vAdam/ConE4/kernel/vAdam/ConE4/bias/vAdam/HidA/kernel/vAdam/HidA/bias/vAdam/HidA2/kernel/vAdam/HidA2/bias/vAdam/HidA3/kernel/vAdam/HidA3/bias/vAdam/HidA4/kernel/vAdam/HidA4/bias/vAdam/HidC/kernel/vAdam/HidC/bias/vAdam/HidC2/kernel/vAdam/HidC2/bias/vAdam/HidC3/kernel/vAdam/HidC3/bias/vAdam/HidC4/kernel/vAdam/HidC4/bias/vAdam/HidE/kernel/vAdam/HidE/bias/vAdam/HidE2/kernel/vAdam/HidE2/bias/vAdam/HidE3/kernel/vAdam/HidE3/bias/vAdam/HidE4/kernel/vAdam/HidE4/bias/vAdam/Hid1/kernel/vAdam/Hid1/bias/vAdam/Hid2/kernel/vAdam/Hid2/bias/vAdam/output/kernel/vAdam/output/bias/v*?
Tin?
?2?*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_232251??)
?
]
A__inference_PoolE_layer_call_and_return_conditional_losses_226657

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????B:0*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????B:02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????0:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?
^
B__inference_PoolA3_layer_call_and_return_conditional_losses_226828

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????!:W S
/
_output_shapes
:?????????!
 
_user_specified_nameinputs
??
?
A__inference_model_layer_call_and_return_conditional_losses_229006	
input%
cone_228833:0
cone_228835:0%
conc_228838:	$
conc_228840:$%
cona_228843:
cona_228845:&
cone2_228854:0
cone2_228856:&
conc2_228859:$
conc2_228861:&
cona2_228864:
cona2_228866:&
cone3_228872:
cone3_228874:&
conc3_228877:
conc3_228879:&
cona3_228882:
cona3_228884:&
cone4_228890:
cone4_228892:&
conc4_228895:
conc4_228897:&
cona4_228900:
cona4_228902: 
hida_228929:???
hida_228931:	?!
hida2_228934:???
hida2_228936:	? 
hida3_228939:
?#?
hida3_228941:	? 
hida4_228944:
??
hida4_228946:	? 
hidc_228949:??	?
hidc_228951:	?!
hidc2_228954:???
hidc2_228956:	? 
hidc3_228959:
? ?
hidc3_228961:	? 
hidc4_228964:
??
hidc4_228966:	? 
hide_228969:???
hide_228971:	?!
hide2_228974:???
hide2_228976:	? 
hide3_228979:
??
hide3_228981:	? 
hide4_228984:
??
hide4_228986:	?
hid1_228990:
??
hid1_228992:	?
hid2_228995:
??
hid2_228997:	? 
output_229000:	?(
output_229002:(
identity??ConA/StatefulPartitionedCall?ConA2/StatefulPartitionedCall?ConA3/StatefulPartitionedCall?ConA4/StatefulPartitionedCall?ConC/StatefulPartitionedCall?ConC2/StatefulPartitionedCall?ConC3/StatefulPartitionedCall?ConC4/StatefulPartitionedCall?ConE/StatefulPartitionedCall?ConE2/StatefulPartitionedCall?ConE3/StatefulPartitionedCall?ConE4/StatefulPartitionedCall?Hid1/StatefulPartitionedCall?Hid2/StatefulPartitionedCall?HidA/StatefulPartitionedCall?HidA2/StatefulPartitionedCall?HidA3/StatefulPartitionedCall?HidA4/StatefulPartitionedCall?HidC/StatefulPartitionedCall?HidC2/StatefulPartitionedCall?HidC3/StatefulPartitionedCall?HidC4/StatefulPartitionedCall?HidE/StatefulPartitionedCall?HidE2/StatefulPartitionedCall?HidE3/StatefulPartitionedCall?HidE4/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?"dropout_10/StatefulPartitionedCall?"dropout_11/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?!dropout_5/StatefulPartitionedCall?!dropout_6/StatefulPartitionedCall?!dropout_7/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?!dropout_9/StatefulPartitionedCall?output/StatefulPartitionedCall?
ConE/StatefulPartitionedCallStatefulPartitionedCallinputcone_228833cone_228835*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConE_layer_call_and_return_conditional_losses_2266132
ConE/StatefulPartitionedCall?
ConC/StatefulPartitionedCallStatefulPartitionedCallinputconc_228838conc_228840*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConC_layer_call_and_return_conditional_losses_2266302
ConC/StatefulPartitionedCall?
ConA/StatefulPartitionedCallStatefulPartitionedCallinputcona_228843cona_228845*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConA_layer_call_and_return_conditional_losses_2266472
ConA/StatefulPartitionedCall?
PoolE/PartitionedCallPartitionedCall%ConE/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????B:0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolE_layer_call_and_return_conditional_losses_2266572
PoolE/PartitionedCall?
PoolC/PartitionedCallPartitionedCall%ConC/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F=$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolC_layer_call_and_return_conditional_losses_2266632
PoolC/PartitionedCall?
PoolA/PartitionedCallPartitionedCall%ConA/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolA_layer_call_and_return_conditional_losses_2266692
PoolA/PartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCallPoolE/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????B:0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2280912#
!dropout_2/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCallPoolC/PartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F=$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_2280682#
!dropout_1/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCallPoolA/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_2280452!
dropout/StatefulPartitionedCall?
ConE2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0cone2_228854cone2_228856*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@8*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConE2_layer_call_and_return_conditional_losses_2267032
ConE2/StatefulPartitionedCall?
ConC2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conc2_228859conc2_228861*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????D;*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConC2_layer_call_and_return_conditional_losses_2267202
ConC2/StatefulPartitionedCall?
ConA2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0cona2_228864cona2_228866*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????G=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConA2_layer_call_and_return_conditional_losses_2267372
ConA2/StatefulPartitionedCall?
PoolE2/PartitionedCallPartitionedCall&ConE2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolE2_layer_call_and_return_conditional_losses_2267472
PoolE2/PartitionedCall?
PoolC2/PartitionedCallPartitionedCall&ConC2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolC2_layer_call_and_return_conditional_losses_2267532
PoolC2/PartitionedCall?
PoolA2/PartitionedCallPartitionedCall&ConA2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA2_layer_call_and_return_conditional_losses_2267592
PoolA2/PartitionedCall?
ConE3/StatefulPartitionedCallStatefulPartitionedCallPoolE2/PartitionedCall:output:0cone3_228872cone3_228874*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConE3_layer_call_and_return_conditional_losses_2267722
ConE3/StatefulPartitionedCall?
ConC3/StatefulPartitionedCallStatefulPartitionedCallPoolC2/PartitionedCall:output:0conc3_228877conc3_228879*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConC3_layer_call_and_return_conditional_losses_2267892
ConC3/StatefulPartitionedCall?
ConA3/StatefulPartitionedCallStatefulPartitionedCallPoolA2/PartitionedCall:output:0cona3_228882cona3_228884*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConA3_layer_call_and_return_conditional_losses_2268062
ConA3/StatefulPartitionedCall?
PoolE3/PartitionedCallPartitionedCall&ConE3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolE3_layer_call_and_return_conditional_losses_2268162
PoolE3/PartitionedCall?
PoolC3/PartitionedCallPartitionedCall&ConC3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolC3_layer_call_and_return_conditional_losses_2268222
PoolC3/PartitionedCall?
PoolA3/PartitionedCallPartitionedCall&ConA3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA3_layer_call_and_return_conditional_losses_2268282
PoolA3/PartitionedCall?
ConE4/StatefulPartitionedCallStatefulPartitionedCallPoolE3/PartitionedCall:output:0cone4_228890cone4_228892*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConE4_layer_call_and_return_conditional_losses_2268412
ConE4/StatefulPartitionedCall?
ConC4/StatefulPartitionedCallStatefulPartitionedCallPoolC3/PartitionedCall:output:0conc4_228895conc4_228897*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConC4_layer_call_and_return_conditional_losses_2268582
ConC4/StatefulPartitionedCall?
ConA4/StatefulPartitionedCallStatefulPartitionedCallPoolA3/PartitionedCall:output:0cona4_228900cona4_228902*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConA4_layer_call_and_return_conditional_losses_2268752
ConA4/StatefulPartitionedCall?
PoolE4/PartitionedCallPartitionedCall&ConE4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolE4_layer_call_and_return_conditional_losses_2268852
PoolE4/PartitionedCall?
PoolC4/PartitionedCallPartitionedCall&ConC4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolC4_layer_call_and_return_conditional_losses_2268912
PoolC4/PartitionedCall?
PoolA4/PartitionedCallPartitionedCall&ConA4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA4_layer_call_and_return_conditional_losses_2268972
PoolA4/PartitionedCall?
"dropout_11/StatefulPartitionedCallStatefulPartitionedCallPoolE4/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_2278872$
"dropout_11/StatefulPartitionedCall?
"dropout_10/StatefulPartitionedCallStatefulPartitionedCallPoolE3/PartitionedCall:output:0#^dropout_11/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_2278642$
"dropout_10/StatefulPartitionedCall?
!dropout_9/StatefulPartitionedCallStatefulPartitionedCallPoolE2/PartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_2278412#
!dropout_9/StatefulPartitionedCall?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCallPoolC4/PartitionedCall:output:0"^dropout_9/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_2278182#
!dropout_8/StatefulPartitionedCall?
!dropout_7/StatefulPartitionedCallStatefulPartitionedCallPoolC3/PartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_2277952#
!dropout_7/StatefulPartitionedCall?
!dropout_6/StatefulPartitionedCallStatefulPartitionedCallPoolC2/PartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_2277722#
!dropout_6/StatefulPartitionedCall?
!dropout_5/StatefulPartitionedCallStatefulPartitionedCallPoolA4/PartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_2277492#
!dropout_5/StatefulPartitionedCall?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCallPoolA3/PartitionedCall:output:0"^dropout_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_2277262#
!dropout_4/StatefulPartitionedCall?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCallPoolA2/PartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2277032#
!dropout_3/StatefulPartitionedCall?
flatten_11/PartitionedCallPartitionedCall+dropout_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_11_layer_call_and_return_conditional_losses_2269682
flatten_11/PartitionedCall?
flatten_10/PartitionedCallPartitionedCall+dropout_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_10_layer_call_and_return_conditional_losses_2269762
flatten_10/PartitionedCall?
flatten_9/PartitionedCallPartitionedCall*dropout_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_9_layer_call_and_return_conditional_losses_2269842
flatten_9/PartitionedCall?
flatten_8/PartitionedCallPartitionedCallPoolE/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_2269922
flatten_8/PartitionedCall?
flatten_7/PartitionedCallPartitionedCall*dropout_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_2270002
flatten_7/PartitionedCall?
flatten_6/PartitionedCallPartitionedCall*dropout_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_2270082
flatten_6/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall*dropout_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_2270162
flatten_5/PartitionedCall?
flatten_4/PartitionedCallPartitionedCallPoolC/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_2270242
flatten_4/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall*dropout_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_2270322
flatten_3/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall*dropout_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_2270402
flatten_2/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall*dropout_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2270482
flatten_1/PartitionedCall?
flatten/PartitionedCallPartitionedCallPoolA/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2270562
flatten/PartitionedCall?
HidA/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0hida_228929hida_228931*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_HidA_layer_call_and_return_conditional_losses_2270692
HidA/StatefulPartitionedCall?
HidA2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0hida2_228934hida2_228936*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidA2_layer_call_and_return_conditional_losses_2270862
HidA2/StatefulPartitionedCall?
HidA3/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0hida3_228939hida3_228941*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidA3_layer_call_and_return_conditional_losses_2271032
HidA3/StatefulPartitionedCall?
HidA4/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0hida4_228944hida4_228946*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidA4_layer_call_and_return_conditional_losses_2271202
HidA4/StatefulPartitionedCall?
HidC/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0hidc_228949hidc_228951*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_HidC_layer_call_and_return_conditional_losses_2271372
HidC/StatefulPartitionedCall?
HidC2/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0hidc2_228954hidc2_228956*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidC2_layer_call_and_return_conditional_losses_2271542
HidC2/StatefulPartitionedCall?
HidC3/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0hidc3_228959hidc3_228961*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidC3_layer_call_and_return_conditional_losses_2271712
HidC3/StatefulPartitionedCall?
HidC4/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0hidc4_228964hidc4_228966*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidC4_layer_call_and_return_conditional_losses_2271882
HidC4/StatefulPartitionedCall?
HidE/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0hide_228969hide_228971*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_HidE_layer_call_and_return_conditional_losses_2272052
HidE/StatefulPartitionedCall?
HidE2/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0hide2_228974hide2_228976*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidE2_layer_call_and_return_conditional_losses_2272222
HidE2/StatefulPartitionedCall?
HidE3/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0hide3_228979hide3_228981*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidE3_layer_call_and_return_conditional_losses_2272392
HidE3/StatefulPartitionedCall?
HidE4/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0hide4_228984hide4_228986*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidE4_layer_call_and_return_conditional_losses_2272562
HidE4/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall%HidA/StatefulPartitionedCall:output:0&HidA2/StatefulPartitionedCall:output:0&HidA3/StatefulPartitionedCall:output:0&HidA4/StatefulPartitionedCall:output:0%HidC/StatefulPartitionedCall:output:0&HidC2/StatefulPartitionedCall:output:0&HidC3/StatefulPartitionedCall:output:0&HidC4/StatefulPartitionedCall:output:0%HidE/StatefulPartitionedCall:output:0&HidE2/StatefulPartitionedCall:output:0&HidE3/StatefulPartitionedCall:output:0&HidE4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_2272792
concatenate/PartitionedCall?
Hid1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0hid1_228990hid1_228992*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_Hid1_layer_call_and_return_conditional_losses_2272922
Hid1/StatefulPartitionedCall?
Hid2/StatefulPartitionedCallStatefulPartitionedCall%Hid1/StatefulPartitionedCall:output:0hid2_228995hid2_228997*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_Hid2_layer_call_and_return_conditional_losses_2273092
Hid2/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall%Hid2/StatefulPartitionedCall:output:0output_229000output_229002*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_2273262 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????(2

Identity?

NoOpNoOp^ConA/StatefulPartitionedCall^ConA2/StatefulPartitionedCall^ConA3/StatefulPartitionedCall^ConA4/StatefulPartitionedCall^ConC/StatefulPartitionedCall^ConC2/StatefulPartitionedCall^ConC3/StatefulPartitionedCall^ConC4/StatefulPartitionedCall^ConE/StatefulPartitionedCall^ConE2/StatefulPartitionedCall^ConE3/StatefulPartitionedCall^ConE4/StatefulPartitionedCall^Hid1/StatefulPartitionedCall^Hid2/StatefulPartitionedCall^HidA/StatefulPartitionedCall^HidA2/StatefulPartitionedCall^HidA3/StatefulPartitionedCall^HidA4/StatefulPartitionedCall^HidC/StatefulPartitionedCall^HidC2/StatefulPartitionedCall^HidC3/StatefulPartitionedCall^HidC4/StatefulPartitionedCall^HidE/StatefulPartitionedCall^HidE2/StatefulPartitionedCall^HidE3/StatefulPartitionedCall^HidE4/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
ConA/StatefulPartitionedCallConA/StatefulPartitionedCall2>
ConA2/StatefulPartitionedCallConA2/StatefulPartitionedCall2>
ConA3/StatefulPartitionedCallConA3/StatefulPartitionedCall2>
ConA4/StatefulPartitionedCallConA4/StatefulPartitionedCall2<
ConC/StatefulPartitionedCallConC/StatefulPartitionedCall2>
ConC2/StatefulPartitionedCallConC2/StatefulPartitionedCall2>
ConC3/StatefulPartitionedCallConC3/StatefulPartitionedCall2>
ConC4/StatefulPartitionedCallConC4/StatefulPartitionedCall2<
ConE/StatefulPartitionedCallConE/StatefulPartitionedCall2>
ConE2/StatefulPartitionedCallConE2/StatefulPartitionedCall2>
ConE3/StatefulPartitionedCallConE3/StatefulPartitionedCall2>
ConE4/StatefulPartitionedCallConE4/StatefulPartitionedCall2<
Hid1/StatefulPartitionedCallHid1/StatefulPartitionedCall2<
Hid2/StatefulPartitionedCallHid2/StatefulPartitionedCall2<
HidA/StatefulPartitionedCallHidA/StatefulPartitionedCall2>
HidA2/StatefulPartitionedCallHidA2/StatefulPartitionedCall2>
HidA3/StatefulPartitionedCallHidA3/StatefulPartitionedCall2>
HidA4/StatefulPartitionedCallHidA4/StatefulPartitionedCall2<
HidC/StatefulPartitionedCallHidC/StatefulPartitionedCall2>
HidC2/StatefulPartitionedCallHidC2/StatefulPartitionedCall2>
HidC3/StatefulPartitionedCallHidC3/StatefulPartitionedCall2>
HidC4/StatefulPartitionedCallHidC4/StatefulPartitionedCall2<
HidE/StatefulPartitionedCallHidE/StatefulPartitionedCall2>
HidE2/StatefulPartitionedCallHidE2/StatefulPartitionedCall2>
HidE3/StatefulPartitionedCallHidE3/StatefulPartitionedCall2>
HidE4/StatefulPartitionedCallHidE4/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
F
*__inference_dropout_2_layer_call_fn_230102

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
:?????????B:0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2266762
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????B:02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????B:0:W S
/
_output_shapes
:?????????B:0
 
_user_specified_nameinputs
?
^
B__inference_PoolA3_layer_call_and_return_conditional_losses_226472

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
C
'__inference_PoolC4_layer_call_fn_230449

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
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolC4_layer_call_and_return_conditional_losses_2265602
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_4_layer_call_and_return_conditional_losses_230538

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_PoolC3_layer_call_and_return_conditional_losses_226494

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_10_layer_call_and_return_conditional_losses_230688

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_8_layer_call_fn_230624

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_2269252
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_PoolE2_layer_call_and_return_conditional_losses_230244

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@8:W S
/
_output_shapes
:?????????@8
 
_user_specified_nameinputs
?
?
'__inference_output_layer_call_fn_231181

inputs
unknown:	?(
	unknown_0:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_2273262
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
@__inference_ConE_layer_call_and_return_conditional_losses_229983

inputs8
conv2d_readvariableop_resource:0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????02
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_230511

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????#2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????#*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????#2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????#2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????#2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????#:W S
/
_output_shapes
:?????????#
 
_user_specified_nameinputs
?
?
@__inference_HidA_layer_call_and_return_conditional_losses_230879

inputs3
matmul_readvariableop_resource:???.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
C
'__inference_PoolE3_layer_call_fn_230354

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolE3_layer_call_and_return_conditional_losses_2268162
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_ConA2_layer_call_fn_230133

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????G=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConA2_layer_call_and_return_conditional_losses_2267372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????G=2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????I?: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????I?
 
_user_specified_nameinputs
?
c
E__inference_dropout_7_layer_call_and_return_conditional_losses_226932

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_ConE_layer_call_fn_229972

inputs!
unknown:0
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConE_layer_call_and_return_conditional_losses_2266132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
%__inference_ConC_layer_call_fn_229952

inputs!
unknown:	$
	unknown_0:$
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConC_layer_call_and_return_conditional_losses_2266302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????$2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_229127	
input!
unknown:0
	unknown_0:0#
	unknown_1:	$
	unknown_2:$#
	unknown_3:
	unknown_4:#
	unknown_5:0
	unknown_6:#
	unknown_7:$
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:$

unknown_15:

unknown_16:$

unknown_17:

unknown_18:$

unknown_19:

unknown_20:$

unknown_21:

unknown_22:

unknown_23:???

unknown_24:	?

unknown_25:???

unknown_26:	?

unknown_27:
?#?

unknown_28:	?

unknown_29:
??

unknown_30:	?

unknown_31:??	?

unknown_32:	?

unknown_33:???

unknown_34:	?

unknown_35:
? ?

unknown_36:	?

unknown_37:
??

unknown_38:	?

unknown_39:???

unknown_40:	?

unknown_41:???

unknown_42:	?

unknown_43:
??

unknown_44:	?

unknown_45:
??

unknown_46:	?

unknown_47:
??

unknown_48:	?

unknown_49:
??

unknown_50:	?

unknown_51:	?(

unknown_52:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_2263312
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
d
E__inference_dropout_5_layer_call_and_return_conditional_losses_230565

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_PoolC2_layer_call_and_return_conditional_losses_226753

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????"*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????"2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????D;:W S
/
_output_shapes
:?????????D;
 
_user_specified_nameinputs
?
C
'__inference_PoolA3_layer_call_fn_230309

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
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA3_layer_call_and_return_conditional_losses_2264722
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
^
B__inference_PoolA2_layer_call_and_return_conditional_losses_230204

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????#*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????G=:W S
/
_output_shapes
:?????????G=
 
_user_specified_nameinputs
?
?
A__inference_ConE3_layer_call_and_return_conditional_losses_226772

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
^
B__inference_PoolE3_layer_call_and_return_conditional_losses_226816

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
A__inference_HidA2_layer_call_and_return_conditional_losses_227086

inputs3
matmul_readvariableop_resource:???.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_6_layer_call_and_return_conditional_losses_230580

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????"2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????"2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????":W S
/
_output_shapes
:?????????"
 
_user_specified_nameinputs
?
?
&__inference_HidA3_layer_call_fn_230908

inputs
unknown:
?#?
	unknown_0:	?
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
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidA3_layer_call_and_return_conditional_losses_2271032
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????#: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????#
 
_user_specified_nameinputs
?
?
A__inference_ConE2_layer_call_and_return_conditional_losses_230184

inputs8
conv2d_readvariableop_resource:0-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@8*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@82	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@82
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@82

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????B:0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????B:0
 
_user_specified_nameinputs
?
?
A__inference_ConC3_layer_call_and_return_conditional_losses_230284

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????": : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????"
 
_user_specified_nameinputs
?
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_228091

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????B:02
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????B:0*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????B:02
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????B:02
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????B:02
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????B:02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????B:0:W S
/
_output_shapes
:?????????B:0
 
_user_specified_nameinputs
?
^
B__inference_PoolA4_layer_call_and_return_conditional_losses_226897

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_230738

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????(? 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????I?:W S
/
_output_shapes
:?????????I?
 
_user_specified_nameinputs
?
]
A__inference_PoolC_layer_call_and_return_conditional_losses_230023

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????F=$*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????F=$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????$:Y U
1
_output_shapes
:???????????$
 
_user_specified_nameinputs
?
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_227703

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????#2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????#*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????#2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????#2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????#2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????#:W S
/
_output_shapes
:?????????#
 
_user_specified_nameinputs
?
C
'__inference_PoolA2_layer_call_fn_230194

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
:?????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA2_layer_call_and_return_conditional_losses_2267592
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????G=:W S
/
_output_shapes
:?????????G=
 
_user_specified_nameinputs
?
?
A__inference_ConC3_layer_call_and_return_conditional_losses_226789

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????": : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????"
 
_user_specified_nameinputs
?
]
A__inference_PoolA_layer_call_and_return_conditional_losses_229998

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
%__inference_ConA_layer_call_fn_229932

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConA_layer_call_and_return_conditional_losses_2266472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
A__inference_ConE3_layer_call_and_return_conditional_losses_230304

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
^
B__inference_PoolA4_layer_call_and_return_conditional_losses_226538

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_9_layer_call_fn_230831

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_9_layer_call_and_return_conditional_losses_2269842
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
A__inference_ConA3_layer_call_and_return_conditional_losses_230264

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????!2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????!2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????#
 
_user_specified_nameinputs
?
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_230112

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????B:02

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????B:02

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????B:0:W S
/
_output_shapes
:?????????B:0
 
_user_specified_nameinputs
?
?
&__inference_HidC2_layer_call_fn_230968

inputs
unknown:???
	unknown_0:	?
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
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidC2_layer_call_and_return_conditional_losses_2271542
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
C
'__inference_PoolC4_layer_call_fn_230454

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolC4_layer_call_and_return_conditional_losses_2268912
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
A__inference_HidC3_layer_call_and_return_conditional_losses_227171

inputs2
matmul_readvariableop_resource:
? ?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
? ?*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_226690

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????I?2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????I?2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????I?:W S
/
_output_shapes
:?????????I?
 
_user_specified_nameinputs
?
^
B__inference_PoolC4_layer_call_and_return_conditional_losses_226891

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
@__inference_Hid2_layer_call_and_return_conditional_losses_227309

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Tanhd
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
A__inference_ConA4_layer_call_and_return_conditional_losses_230384

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_PoolA2_layer_call_and_return_conditional_losses_226759

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????#*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????G=:W S
/
_output_shapes
:?????????G=
 
_user_specified_nameinputs
?
d
E__inference_dropout_7_layer_call_and_return_conditional_losses_230619

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_226946

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_HidC3_layer_call_fn_230988

inputs
unknown:
? ?
	unknown_0:	?
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
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidC3_layer_call_and_return_conditional_losses_2271712
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
C
'__inference_PoolA2_layer_call_fn_230189

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
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA2_layer_call_and_return_conditional_losses_2264062
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_flatten_10_layer_call_and_return_conditional_losses_230848

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????<  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_10_layer_call_and_return_conditional_losses_226911

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
A__inference_ConC2_layer_call_and_return_conditional_losses_230164

inputs8
conv2d_readvariableop_resource:$-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????D;*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????D;2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????D;2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????D;2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????F=$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????F=$
 
_user_specified_nameinputs
?
?
&__inference_HidE3_layer_call_fn_231068

inputs
unknown:
??
	unknown_0:	?
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
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidE3_layer_call_and_return_conditional_losses_2272392
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_6_layer_call_and_return_conditional_losses_230592

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????"2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????"*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????"2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????"2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????"2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????"2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????":W S
/
_output_shapes
:?????????"
 
_user_specified_nameinputs
?
?
&__inference_ConA3_layer_call_fn_230253

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConA3_layer_call_and_return_conditional_losses_2268062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????!2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????#: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????#
 
_user_specified_nameinputs
?
F
*__inference_flatten_2_layer_call_fn_230754

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
:??????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_2270402
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_HidA_layer_call_fn_230868

inputs
unknown:???
	unknown_0:	?
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
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_HidA_layer_call_and_return_conditional_losses_2270692
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
A__inference_HidA3_layer_call_and_return_conditional_losses_230919

inputs2
matmul_readvariableop_resource:
?#?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?#?*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????#
 
_user_specified_nameinputs
??
?
A__inference_model_layer_call_and_return_conditional_losses_228830	
input%
cone_228657:0
cone_228659:0%
conc_228662:	$
conc_228664:$%
cona_228667:
cona_228669:&
cone2_228678:0
cone2_228680:&
conc2_228683:$
conc2_228685:&
cona2_228688:
cona2_228690:&
cone3_228696:
cone3_228698:&
conc3_228701:
conc3_228703:&
cona3_228706:
cona3_228708:&
cone4_228714:
cone4_228716:&
conc4_228719:
conc4_228721:&
cona4_228724:
cona4_228726: 
hida_228753:???
hida_228755:	?!
hida2_228758:???
hida2_228760:	? 
hida3_228763:
?#?
hida3_228765:	? 
hida4_228768:
??
hida4_228770:	? 
hidc_228773:??	?
hidc_228775:	?!
hidc2_228778:???
hidc2_228780:	? 
hidc3_228783:
? ?
hidc3_228785:	? 
hidc4_228788:
??
hidc4_228790:	? 
hide_228793:???
hide_228795:	?!
hide2_228798:???
hide2_228800:	? 
hide3_228803:
??
hide3_228805:	? 
hide4_228808:
??
hide4_228810:	?
hid1_228814:
??
hid1_228816:	?
hid2_228819:
??
hid2_228821:	? 
output_228824:	?(
output_228826:(
identity??ConA/StatefulPartitionedCall?ConA2/StatefulPartitionedCall?ConA3/StatefulPartitionedCall?ConA4/StatefulPartitionedCall?ConC/StatefulPartitionedCall?ConC2/StatefulPartitionedCall?ConC3/StatefulPartitionedCall?ConC4/StatefulPartitionedCall?ConE/StatefulPartitionedCall?ConE2/StatefulPartitionedCall?ConE3/StatefulPartitionedCall?ConE4/StatefulPartitionedCall?Hid1/StatefulPartitionedCall?Hid2/StatefulPartitionedCall?HidA/StatefulPartitionedCall?HidA2/StatefulPartitionedCall?HidA3/StatefulPartitionedCall?HidA4/StatefulPartitionedCall?HidC/StatefulPartitionedCall?HidC2/StatefulPartitionedCall?HidC3/StatefulPartitionedCall?HidC4/StatefulPartitionedCall?HidE/StatefulPartitionedCall?HidE2/StatefulPartitionedCall?HidE3/StatefulPartitionedCall?HidE4/StatefulPartitionedCall?output/StatefulPartitionedCall?
ConE/StatefulPartitionedCallStatefulPartitionedCallinputcone_228657cone_228659*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConE_layer_call_and_return_conditional_losses_2266132
ConE/StatefulPartitionedCall?
ConC/StatefulPartitionedCallStatefulPartitionedCallinputconc_228662conc_228664*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConC_layer_call_and_return_conditional_losses_2266302
ConC/StatefulPartitionedCall?
ConA/StatefulPartitionedCallStatefulPartitionedCallinputcona_228667cona_228669*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConA_layer_call_and_return_conditional_losses_2266472
ConA/StatefulPartitionedCall?
PoolE/PartitionedCallPartitionedCall%ConE/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????B:0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolE_layer_call_and_return_conditional_losses_2266572
PoolE/PartitionedCall?
PoolC/PartitionedCallPartitionedCall%ConC/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F=$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolC_layer_call_and_return_conditional_losses_2266632
PoolC/PartitionedCall?
PoolA/PartitionedCallPartitionedCall%ConA/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolA_layer_call_and_return_conditional_losses_2266692
PoolA/PartitionedCall?
dropout_2/PartitionedCallPartitionedCallPoolE/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????B:0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2266762
dropout_2/PartitionedCall?
dropout_1/PartitionedCallPartitionedCallPoolC/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F=$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_2266832
dropout_1/PartitionedCall?
dropout/PartitionedCallPartitionedCallPoolA/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_2266902
dropout/PartitionedCall?
ConE2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0cone2_228678cone2_228680*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@8*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConE2_layer_call_and_return_conditional_losses_2267032
ConE2/StatefulPartitionedCall?
ConC2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conc2_228683conc2_228685*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????D;*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConC2_layer_call_and_return_conditional_losses_2267202
ConC2/StatefulPartitionedCall?
ConA2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0cona2_228688cona2_228690*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????G=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConA2_layer_call_and_return_conditional_losses_2267372
ConA2/StatefulPartitionedCall?
PoolE2/PartitionedCallPartitionedCall&ConE2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolE2_layer_call_and_return_conditional_losses_2267472
PoolE2/PartitionedCall?
PoolC2/PartitionedCallPartitionedCall&ConC2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolC2_layer_call_and_return_conditional_losses_2267532
PoolC2/PartitionedCall?
PoolA2/PartitionedCallPartitionedCall&ConA2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA2_layer_call_and_return_conditional_losses_2267592
PoolA2/PartitionedCall?
ConE3/StatefulPartitionedCallStatefulPartitionedCallPoolE2/PartitionedCall:output:0cone3_228696cone3_228698*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConE3_layer_call_and_return_conditional_losses_2267722
ConE3/StatefulPartitionedCall?
ConC3/StatefulPartitionedCallStatefulPartitionedCallPoolC2/PartitionedCall:output:0conc3_228701conc3_228703*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConC3_layer_call_and_return_conditional_losses_2267892
ConC3/StatefulPartitionedCall?
ConA3/StatefulPartitionedCallStatefulPartitionedCallPoolA2/PartitionedCall:output:0cona3_228706cona3_228708*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConA3_layer_call_and_return_conditional_losses_2268062
ConA3/StatefulPartitionedCall?
PoolE3/PartitionedCallPartitionedCall&ConE3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolE3_layer_call_and_return_conditional_losses_2268162
PoolE3/PartitionedCall?
PoolC3/PartitionedCallPartitionedCall&ConC3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolC3_layer_call_and_return_conditional_losses_2268222
PoolC3/PartitionedCall?
PoolA3/PartitionedCallPartitionedCall&ConA3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA3_layer_call_and_return_conditional_losses_2268282
PoolA3/PartitionedCall?
ConE4/StatefulPartitionedCallStatefulPartitionedCallPoolE3/PartitionedCall:output:0cone4_228714cone4_228716*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConE4_layer_call_and_return_conditional_losses_2268412
ConE4/StatefulPartitionedCall?
ConC4/StatefulPartitionedCallStatefulPartitionedCallPoolC3/PartitionedCall:output:0conc4_228719conc4_228721*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConC4_layer_call_and_return_conditional_losses_2268582
ConC4/StatefulPartitionedCall?
ConA4/StatefulPartitionedCallStatefulPartitionedCallPoolA3/PartitionedCall:output:0cona4_228724cona4_228726*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConA4_layer_call_and_return_conditional_losses_2268752
ConA4/StatefulPartitionedCall?
PoolE4/PartitionedCallPartitionedCall&ConE4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolE4_layer_call_and_return_conditional_losses_2268852
PoolE4/PartitionedCall?
PoolC4/PartitionedCallPartitionedCall&ConC4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolC4_layer_call_and_return_conditional_losses_2268912
PoolC4/PartitionedCall?
PoolA4/PartitionedCallPartitionedCall&ConA4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA4_layer_call_and_return_conditional_losses_2268972
PoolA4/PartitionedCall?
dropout_11/PartitionedCallPartitionedCallPoolE4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_2269042
dropout_11/PartitionedCall?
dropout_10/PartitionedCallPartitionedCallPoolE3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_2269112
dropout_10/PartitionedCall?
dropout_9/PartitionedCallPartitionedCallPoolE2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_2269182
dropout_9/PartitionedCall?
dropout_8/PartitionedCallPartitionedCallPoolC4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_2269252
dropout_8/PartitionedCall?
dropout_7/PartitionedCallPartitionedCallPoolC3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_2269322
dropout_7/PartitionedCall?
dropout_6/PartitionedCallPartitionedCallPoolC2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_2269392
dropout_6/PartitionedCall?
dropout_5/PartitionedCallPartitionedCallPoolA4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_2269462
dropout_5/PartitionedCall?
dropout_4/PartitionedCallPartitionedCallPoolA3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_2269532
dropout_4/PartitionedCall?
dropout_3/PartitionedCallPartitionedCallPoolA2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2269602
dropout_3/PartitionedCall?
flatten_11/PartitionedCallPartitionedCall#dropout_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_11_layer_call_and_return_conditional_losses_2269682
flatten_11/PartitionedCall?
flatten_10/PartitionedCallPartitionedCall#dropout_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_10_layer_call_and_return_conditional_losses_2269762
flatten_10/PartitionedCall?
flatten_9/PartitionedCallPartitionedCall"dropout_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_9_layer_call_and_return_conditional_losses_2269842
flatten_9/PartitionedCall?
flatten_8/PartitionedCallPartitionedCallPoolE/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_2269922
flatten_8/PartitionedCall?
flatten_7/PartitionedCallPartitionedCall"dropout_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_2270002
flatten_7/PartitionedCall?
flatten_6/PartitionedCallPartitionedCall"dropout_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_2270082
flatten_6/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall"dropout_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_2270162
flatten_5/PartitionedCall?
flatten_4/PartitionedCallPartitionedCallPoolC/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_2270242
flatten_4/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall"dropout_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_2270322
flatten_3/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall"dropout_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_2270402
flatten_2/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall"dropout_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2270482
flatten_1/PartitionedCall?
flatten/PartitionedCallPartitionedCallPoolA/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2270562
flatten/PartitionedCall?
HidA/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0hida_228753hida_228755*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_HidA_layer_call_and_return_conditional_losses_2270692
HidA/StatefulPartitionedCall?
HidA2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0hida2_228758hida2_228760*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidA2_layer_call_and_return_conditional_losses_2270862
HidA2/StatefulPartitionedCall?
HidA3/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0hida3_228763hida3_228765*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidA3_layer_call_and_return_conditional_losses_2271032
HidA3/StatefulPartitionedCall?
HidA4/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0hida4_228768hida4_228770*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidA4_layer_call_and_return_conditional_losses_2271202
HidA4/StatefulPartitionedCall?
HidC/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0hidc_228773hidc_228775*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_HidC_layer_call_and_return_conditional_losses_2271372
HidC/StatefulPartitionedCall?
HidC2/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0hidc2_228778hidc2_228780*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidC2_layer_call_and_return_conditional_losses_2271542
HidC2/StatefulPartitionedCall?
HidC3/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0hidc3_228783hidc3_228785*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidC3_layer_call_and_return_conditional_losses_2271712
HidC3/StatefulPartitionedCall?
HidC4/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0hidc4_228788hidc4_228790*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidC4_layer_call_and_return_conditional_losses_2271882
HidC4/StatefulPartitionedCall?
HidE/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0hide_228793hide_228795*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_HidE_layer_call_and_return_conditional_losses_2272052
HidE/StatefulPartitionedCall?
HidE2/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0hide2_228798hide2_228800*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidE2_layer_call_and_return_conditional_losses_2272222
HidE2/StatefulPartitionedCall?
HidE3/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0hide3_228803hide3_228805*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidE3_layer_call_and_return_conditional_losses_2272392
HidE3/StatefulPartitionedCall?
HidE4/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0hide4_228808hide4_228810*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidE4_layer_call_and_return_conditional_losses_2272562
HidE4/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall%HidA/StatefulPartitionedCall:output:0&HidA2/StatefulPartitionedCall:output:0&HidA3/StatefulPartitionedCall:output:0&HidA4/StatefulPartitionedCall:output:0%HidC/StatefulPartitionedCall:output:0&HidC2/StatefulPartitionedCall:output:0&HidC3/StatefulPartitionedCall:output:0&HidC4/StatefulPartitionedCall:output:0%HidE/StatefulPartitionedCall:output:0&HidE2/StatefulPartitionedCall:output:0&HidE3/StatefulPartitionedCall:output:0&HidE4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_2272792
concatenate/PartitionedCall?
Hid1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0hid1_228814hid1_228816*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_Hid1_layer_call_and_return_conditional_losses_2272922
Hid1/StatefulPartitionedCall?
Hid2/StatefulPartitionedCallStatefulPartitionedCall%Hid1/StatefulPartitionedCall:output:0hid2_228819hid2_228821*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_Hid2_layer_call_and_return_conditional_losses_2273092
Hid2/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall%Hid2/StatefulPartitionedCall:output:0output_228824output_228826*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_2273262 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????(2

Identity?
NoOpNoOp^ConA/StatefulPartitionedCall^ConA2/StatefulPartitionedCall^ConA3/StatefulPartitionedCall^ConA4/StatefulPartitionedCall^ConC/StatefulPartitionedCall^ConC2/StatefulPartitionedCall^ConC3/StatefulPartitionedCall^ConC4/StatefulPartitionedCall^ConE/StatefulPartitionedCall^ConE2/StatefulPartitionedCall^ConE3/StatefulPartitionedCall^ConE4/StatefulPartitionedCall^Hid1/StatefulPartitionedCall^Hid2/StatefulPartitionedCall^HidA/StatefulPartitionedCall^HidA2/StatefulPartitionedCall^HidA3/StatefulPartitionedCall^HidA4/StatefulPartitionedCall^HidC/StatefulPartitionedCall^HidC2/StatefulPartitionedCall^HidC3/StatefulPartitionedCall^HidC4/StatefulPartitionedCall^HidE/StatefulPartitionedCall^HidE2/StatefulPartitionedCall^HidE3/StatefulPartitionedCall^HidE4/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
ConA/StatefulPartitionedCallConA/StatefulPartitionedCall2>
ConA2/StatefulPartitionedCallConA2/StatefulPartitionedCall2>
ConA3/StatefulPartitionedCallConA3/StatefulPartitionedCall2>
ConA4/StatefulPartitionedCallConA4/StatefulPartitionedCall2<
ConC/StatefulPartitionedCallConC/StatefulPartitionedCall2>
ConC2/StatefulPartitionedCallConC2/StatefulPartitionedCall2>
ConC3/StatefulPartitionedCallConC3/StatefulPartitionedCall2>
ConC4/StatefulPartitionedCallConC4/StatefulPartitionedCall2<
ConE/StatefulPartitionedCallConE/StatefulPartitionedCall2>
ConE2/StatefulPartitionedCallConE2/StatefulPartitionedCall2>
ConE3/StatefulPartitionedCallConE3/StatefulPartitionedCall2>
ConE4/StatefulPartitionedCallConE4/StatefulPartitionedCall2<
Hid1/StatefulPartitionedCallHid1/StatefulPartitionedCall2<
Hid2/StatefulPartitionedCallHid2/StatefulPartitionedCall2<
HidA/StatefulPartitionedCallHidA/StatefulPartitionedCall2>
HidA2/StatefulPartitionedCallHidA2/StatefulPartitionedCall2>
HidA3/StatefulPartitionedCallHidA3/StatefulPartitionedCall2>
HidA4/StatefulPartitionedCallHidA4/StatefulPartitionedCall2<
HidC/StatefulPartitionedCallHidC/StatefulPartitionedCall2>
HidC2/StatefulPartitionedCallHidC2/StatefulPartitionedCall2>
HidC3/StatefulPartitionedCallHidC3/StatefulPartitionedCall2>
HidC4/StatefulPartitionedCallHidC4/StatefulPartitionedCall2<
HidE/StatefulPartitionedCallHidE/StatefulPartitionedCall2>
HidE2/StatefulPartitionedCallHidE2/StatefulPartitionedCall2>
HidE3/StatefulPartitionedCallHidE3/StatefulPartitionedCall2>
HidE4/StatefulPartitionedCallHidE4/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
b
C__inference_dropout_layer_call_and_return_conditional_losses_228045

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????I?2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????I?*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????I?2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????I?2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????I?2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????I?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????I?:W S
/
_output_shapes
:?????????I?
 
_user_specified_nameinputs
?
B
&__inference_PoolE_layer_call_fn_230028

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
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolE_layer_call_and_return_conditional_losses_2263842
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_226683

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????F=$2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????F=$2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????F=$:W S
/
_output_shapes
:?????????F=$
 
_user_specified_nameinputs
?
?
A__inference_ConA2_layer_call_and_return_conditional_losses_230144

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????G=*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????G=2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????G=2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????G=2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????I?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????I?
 
_user_specified_nameinputs
?
^
B__inference_PoolC4_layer_call_and_return_conditional_losses_230464

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_HidC_layer_call_fn_230948

inputs
unknown:??	?
	unknown_0:	?
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
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_HidC_layer_call_and_return_conditional_losses_2271372
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????	: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????	
 
_user_specified_nameinputs
?
?
@__inference_ConC_layer_call_and_return_conditional_losses_226630

inputs8
conv2d_readvariableop_resource:	$-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	$*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????$2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????$2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
A__inference_model_layer_call_and_return_conditional_losses_228430

inputs%
cone_228257:0
cone_228259:0%
conc_228262:	$
conc_228264:$%
cona_228267:
cona_228269:&
cone2_228278:0
cone2_228280:&
conc2_228283:$
conc2_228285:&
cona2_228288:
cona2_228290:&
cone3_228296:
cone3_228298:&
conc3_228301:
conc3_228303:&
cona3_228306:
cona3_228308:&
cone4_228314:
cone4_228316:&
conc4_228319:
conc4_228321:&
cona4_228324:
cona4_228326: 
hida_228353:???
hida_228355:	?!
hida2_228358:???
hida2_228360:	? 
hida3_228363:
?#?
hida3_228365:	? 
hida4_228368:
??
hida4_228370:	? 
hidc_228373:??	?
hidc_228375:	?!
hidc2_228378:???
hidc2_228380:	? 
hidc3_228383:
? ?
hidc3_228385:	? 
hidc4_228388:
??
hidc4_228390:	? 
hide_228393:???
hide_228395:	?!
hide2_228398:???
hide2_228400:	? 
hide3_228403:
??
hide3_228405:	? 
hide4_228408:
??
hide4_228410:	?
hid1_228414:
??
hid1_228416:	?
hid2_228419:
??
hid2_228421:	? 
output_228424:	?(
output_228426:(
identity??ConA/StatefulPartitionedCall?ConA2/StatefulPartitionedCall?ConA3/StatefulPartitionedCall?ConA4/StatefulPartitionedCall?ConC/StatefulPartitionedCall?ConC2/StatefulPartitionedCall?ConC3/StatefulPartitionedCall?ConC4/StatefulPartitionedCall?ConE/StatefulPartitionedCall?ConE2/StatefulPartitionedCall?ConE3/StatefulPartitionedCall?ConE4/StatefulPartitionedCall?Hid1/StatefulPartitionedCall?Hid2/StatefulPartitionedCall?HidA/StatefulPartitionedCall?HidA2/StatefulPartitionedCall?HidA3/StatefulPartitionedCall?HidA4/StatefulPartitionedCall?HidC/StatefulPartitionedCall?HidC2/StatefulPartitionedCall?HidC3/StatefulPartitionedCall?HidC4/StatefulPartitionedCall?HidE/StatefulPartitionedCall?HidE2/StatefulPartitionedCall?HidE3/StatefulPartitionedCall?HidE4/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?"dropout_10/StatefulPartitionedCall?"dropout_11/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?!dropout_5/StatefulPartitionedCall?!dropout_6/StatefulPartitionedCall?!dropout_7/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?!dropout_9/StatefulPartitionedCall?output/StatefulPartitionedCall?
ConE/StatefulPartitionedCallStatefulPartitionedCallinputscone_228257cone_228259*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConE_layer_call_and_return_conditional_losses_2266132
ConE/StatefulPartitionedCall?
ConC/StatefulPartitionedCallStatefulPartitionedCallinputsconc_228262conc_228264*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConC_layer_call_and_return_conditional_losses_2266302
ConC/StatefulPartitionedCall?
ConA/StatefulPartitionedCallStatefulPartitionedCallinputscona_228267cona_228269*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConA_layer_call_and_return_conditional_losses_2266472
ConA/StatefulPartitionedCall?
PoolE/PartitionedCallPartitionedCall%ConE/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????B:0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolE_layer_call_and_return_conditional_losses_2266572
PoolE/PartitionedCall?
PoolC/PartitionedCallPartitionedCall%ConC/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F=$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolC_layer_call_and_return_conditional_losses_2266632
PoolC/PartitionedCall?
PoolA/PartitionedCallPartitionedCall%ConA/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolA_layer_call_and_return_conditional_losses_2266692
PoolA/PartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCallPoolE/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????B:0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2280912#
!dropout_2/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCallPoolC/PartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F=$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_2280682#
!dropout_1/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCallPoolA/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_2280452!
dropout/StatefulPartitionedCall?
ConE2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0cone2_228278cone2_228280*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@8*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConE2_layer_call_and_return_conditional_losses_2267032
ConE2/StatefulPartitionedCall?
ConC2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conc2_228283conc2_228285*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????D;*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConC2_layer_call_and_return_conditional_losses_2267202
ConC2/StatefulPartitionedCall?
ConA2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0cona2_228288cona2_228290*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????G=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConA2_layer_call_and_return_conditional_losses_2267372
ConA2/StatefulPartitionedCall?
PoolE2/PartitionedCallPartitionedCall&ConE2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolE2_layer_call_and_return_conditional_losses_2267472
PoolE2/PartitionedCall?
PoolC2/PartitionedCallPartitionedCall&ConC2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolC2_layer_call_and_return_conditional_losses_2267532
PoolC2/PartitionedCall?
PoolA2/PartitionedCallPartitionedCall&ConA2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA2_layer_call_and_return_conditional_losses_2267592
PoolA2/PartitionedCall?
ConE3/StatefulPartitionedCallStatefulPartitionedCallPoolE2/PartitionedCall:output:0cone3_228296cone3_228298*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConE3_layer_call_and_return_conditional_losses_2267722
ConE3/StatefulPartitionedCall?
ConC3/StatefulPartitionedCallStatefulPartitionedCallPoolC2/PartitionedCall:output:0conc3_228301conc3_228303*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConC3_layer_call_and_return_conditional_losses_2267892
ConC3/StatefulPartitionedCall?
ConA3/StatefulPartitionedCallStatefulPartitionedCallPoolA2/PartitionedCall:output:0cona3_228306cona3_228308*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConA3_layer_call_and_return_conditional_losses_2268062
ConA3/StatefulPartitionedCall?
PoolE3/PartitionedCallPartitionedCall&ConE3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolE3_layer_call_and_return_conditional_losses_2268162
PoolE3/PartitionedCall?
PoolC3/PartitionedCallPartitionedCall&ConC3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolC3_layer_call_and_return_conditional_losses_2268222
PoolC3/PartitionedCall?
PoolA3/PartitionedCallPartitionedCall&ConA3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA3_layer_call_and_return_conditional_losses_2268282
PoolA3/PartitionedCall?
ConE4/StatefulPartitionedCallStatefulPartitionedCallPoolE3/PartitionedCall:output:0cone4_228314cone4_228316*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConE4_layer_call_and_return_conditional_losses_2268412
ConE4/StatefulPartitionedCall?
ConC4/StatefulPartitionedCallStatefulPartitionedCallPoolC3/PartitionedCall:output:0conc4_228319conc4_228321*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConC4_layer_call_and_return_conditional_losses_2268582
ConC4/StatefulPartitionedCall?
ConA4/StatefulPartitionedCallStatefulPartitionedCallPoolA3/PartitionedCall:output:0cona4_228324cona4_228326*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConA4_layer_call_and_return_conditional_losses_2268752
ConA4/StatefulPartitionedCall?
PoolE4/PartitionedCallPartitionedCall&ConE4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolE4_layer_call_and_return_conditional_losses_2268852
PoolE4/PartitionedCall?
PoolC4/PartitionedCallPartitionedCall&ConC4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolC4_layer_call_and_return_conditional_losses_2268912
PoolC4/PartitionedCall?
PoolA4/PartitionedCallPartitionedCall&ConA4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA4_layer_call_and_return_conditional_losses_2268972
PoolA4/PartitionedCall?
"dropout_11/StatefulPartitionedCallStatefulPartitionedCallPoolE4/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_2278872$
"dropout_11/StatefulPartitionedCall?
"dropout_10/StatefulPartitionedCallStatefulPartitionedCallPoolE3/PartitionedCall:output:0#^dropout_11/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_2278642$
"dropout_10/StatefulPartitionedCall?
!dropout_9/StatefulPartitionedCallStatefulPartitionedCallPoolE2/PartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_2278412#
!dropout_9/StatefulPartitionedCall?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCallPoolC4/PartitionedCall:output:0"^dropout_9/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_2278182#
!dropout_8/StatefulPartitionedCall?
!dropout_7/StatefulPartitionedCallStatefulPartitionedCallPoolC3/PartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_2277952#
!dropout_7/StatefulPartitionedCall?
!dropout_6/StatefulPartitionedCallStatefulPartitionedCallPoolC2/PartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_2277722#
!dropout_6/StatefulPartitionedCall?
!dropout_5/StatefulPartitionedCallStatefulPartitionedCallPoolA4/PartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_2277492#
!dropout_5/StatefulPartitionedCall?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCallPoolA3/PartitionedCall:output:0"^dropout_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_2277262#
!dropout_4/StatefulPartitionedCall?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCallPoolA2/PartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2277032#
!dropout_3/StatefulPartitionedCall?
flatten_11/PartitionedCallPartitionedCall+dropout_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_11_layer_call_and_return_conditional_losses_2269682
flatten_11/PartitionedCall?
flatten_10/PartitionedCallPartitionedCall+dropout_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_10_layer_call_and_return_conditional_losses_2269762
flatten_10/PartitionedCall?
flatten_9/PartitionedCallPartitionedCall*dropout_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_9_layer_call_and_return_conditional_losses_2269842
flatten_9/PartitionedCall?
flatten_8/PartitionedCallPartitionedCallPoolE/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_2269922
flatten_8/PartitionedCall?
flatten_7/PartitionedCallPartitionedCall*dropout_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_2270002
flatten_7/PartitionedCall?
flatten_6/PartitionedCallPartitionedCall*dropout_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_2270082
flatten_6/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall*dropout_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_2270162
flatten_5/PartitionedCall?
flatten_4/PartitionedCallPartitionedCallPoolC/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_2270242
flatten_4/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall*dropout_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_2270322
flatten_3/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall*dropout_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_2270402
flatten_2/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall*dropout_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2270482
flatten_1/PartitionedCall?
flatten/PartitionedCallPartitionedCallPoolA/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2270562
flatten/PartitionedCall?
HidA/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0hida_228353hida_228355*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_HidA_layer_call_and_return_conditional_losses_2270692
HidA/StatefulPartitionedCall?
HidA2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0hida2_228358hida2_228360*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidA2_layer_call_and_return_conditional_losses_2270862
HidA2/StatefulPartitionedCall?
HidA3/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0hida3_228363hida3_228365*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidA3_layer_call_and_return_conditional_losses_2271032
HidA3/StatefulPartitionedCall?
HidA4/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0hida4_228368hida4_228370*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidA4_layer_call_and_return_conditional_losses_2271202
HidA4/StatefulPartitionedCall?
HidC/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0hidc_228373hidc_228375*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_HidC_layer_call_and_return_conditional_losses_2271372
HidC/StatefulPartitionedCall?
HidC2/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0hidc2_228378hidc2_228380*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidC2_layer_call_and_return_conditional_losses_2271542
HidC2/StatefulPartitionedCall?
HidC3/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0hidc3_228383hidc3_228385*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidC3_layer_call_and_return_conditional_losses_2271712
HidC3/StatefulPartitionedCall?
HidC4/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0hidc4_228388hidc4_228390*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidC4_layer_call_and_return_conditional_losses_2271882
HidC4/StatefulPartitionedCall?
HidE/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0hide_228393hide_228395*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_HidE_layer_call_and_return_conditional_losses_2272052
HidE/StatefulPartitionedCall?
HidE2/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0hide2_228398hide2_228400*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidE2_layer_call_and_return_conditional_losses_2272222
HidE2/StatefulPartitionedCall?
HidE3/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0hide3_228403hide3_228405*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidE3_layer_call_and_return_conditional_losses_2272392
HidE3/StatefulPartitionedCall?
HidE4/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0hide4_228408hide4_228410*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidE4_layer_call_and_return_conditional_losses_2272562
HidE4/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall%HidA/StatefulPartitionedCall:output:0&HidA2/StatefulPartitionedCall:output:0&HidA3/StatefulPartitionedCall:output:0&HidA4/StatefulPartitionedCall:output:0%HidC/StatefulPartitionedCall:output:0&HidC2/StatefulPartitionedCall:output:0&HidC3/StatefulPartitionedCall:output:0&HidC4/StatefulPartitionedCall:output:0%HidE/StatefulPartitionedCall:output:0&HidE2/StatefulPartitionedCall:output:0&HidE3/StatefulPartitionedCall:output:0&HidE4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_2272792
concatenate/PartitionedCall?
Hid1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0hid1_228414hid1_228416*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_Hid1_layer_call_and_return_conditional_losses_2272922
Hid1/StatefulPartitionedCall?
Hid2/StatefulPartitionedCallStatefulPartitionedCall%Hid1/StatefulPartitionedCall:output:0hid2_228419hid2_228421*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_Hid2_layer_call_and_return_conditional_losses_2273092
Hid2/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall%Hid2/StatefulPartitionedCall:output:0output_228424output_228426*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_2273262 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????(2

Identity?

NoOpNoOp^ConA/StatefulPartitionedCall^ConA2/StatefulPartitionedCall^ConA3/StatefulPartitionedCall^ConA4/StatefulPartitionedCall^ConC/StatefulPartitionedCall^ConC2/StatefulPartitionedCall^ConC3/StatefulPartitionedCall^ConC4/StatefulPartitionedCall^ConE/StatefulPartitionedCall^ConE2/StatefulPartitionedCall^ConE3/StatefulPartitionedCall^ConE4/StatefulPartitionedCall^Hid1/StatefulPartitionedCall^Hid2/StatefulPartitionedCall^HidA/StatefulPartitionedCall^HidA2/StatefulPartitionedCall^HidA3/StatefulPartitionedCall^HidA4/StatefulPartitionedCall^HidC/StatefulPartitionedCall^HidC2/StatefulPartitionedCall^HidC3/StatefulPartitionedCall^HidC4/StatefulPartitionedCall^HidE/StatefulPartitionedCall^HidE2/StatefulPartitionedCall^HidE3/StatefulPartitionedCall^HidE4/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
ConA/StatefulPartitionedCallConA/StatefulPartitionedCall2>
ConA2/StatefulPartitionedCallConA2/StatefulPartitionedCall2>
ConA3/StatefulPartitionedCallConA3/StatefulPartitionedCall2>
ConA4/StatefulPartitionedCallConA4/StatefulPartitionedCall2<
ConC/StatefulPartitionedCallConC/StatefulPartitionedCall2>
ConC2/StatefulPartitionedCallConC2/StatefulPartitionedCall2>
ConC3/StatefulPartitionedCallConC3/StatefulPartitionedCall2>
ConC4/StatefulPartitionedCallConC4/StatefulPartitionedCall2<
ConE/StatefulPartitionedCallConE/StatefulPartitionedCall2>
ConE2/StatefulPartitionedCallConE2/StatefulPartitionedCall2>
ConE3/StatefulPartitionedCallConE3/StatefulPartitionedCall2>
ConE4/StatefulPartitionedCallConE4/StatefulPartitionedCall2<
Hid1/StatefulPartitionedCallHid1/StatefulPartitionedCall2<
Hid2/StatefulPartitionedCallHid2/StatefulPartitionedCall2<
HidA/StatefulPartitionedCallHidA/StatefulPartitionedCall2>
HidA2/StatefulPartitionedCallHidA2/StatefulPartitionedCall2>
HidA3/StatefulPartitionedCallHidA3/StatefulPartitionedCall2>
HidA4/StatefulPartitionedCallHidA4/StatefulPartitionedCall2<
HidC/StatefulPartitionedCallHidC/StatefulPartitionedCall2>
HidC2/StatefulPartitionedCallHidC2/StatefulPartitionedCall2>
HidC3/StatefulPartitionedCallHidC3/StatefulPartitionedCall2>
HidC4/StatefulPartitionedCallHidC4/StatefulPartitionedCall2<
HidE/StatefulPartitionedCallHidE/StatefulPartitionedCall2>
HidE2/StatefulPartitionedCallHidE2/StatefulPartitionedCall2>
HidE3/StatefulPartitionedCallHidE3/StatefulPartitionedCall2>
HidE4/StatefulPartitionedCallHidE4/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
]
A__inference_PoolE_layer_call_and_return_conditional_losses_230043

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????B:0*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????B:02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????0:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?
^
B__inference_PoolE3_layer_call_and_return_conditional_losses_226516

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
^
B__inference_PoolC3_layer_call_and_return_conditional_losses_230344

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_227040

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????#2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_228068

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????F=$2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????F=$*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????F=$2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????F=$2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????F=$2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????F=$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????F=$:W S
/
_output_shapes
:?????????F=$
 
_user_specified_nameinputs
?
?
@__inference_HidE_layer_call_and_return_conditional_losses_227205

inputs3
matmul_readvariableop_resource:???.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_5_layer_call_fn_230548

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_2277492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_9_layer_call_fn_230656

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_2278412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
A__inference_ConA2_layer_call_and_return_conditional_losses_226737

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????G=*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????G=2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????G=2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????G=2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????I?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????I?
 
_user_specified_nameinputs
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_227056

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????(? 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????I?:W S
/
_output_shapes
:?????????I?
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_229353

inputs!
unknown:0
	unknown_0:0#
	unknown_1:	$
	unknown_2:$#
	unknown_3:
	unknown_4:#
	unknown_5:0
	unknown_6:#
	unknown_7:$
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:$

unknown_15:

unknown_16:$

unknown_17:

unknown_18:$

unknown_19:

unknown_20:$

unknown_21:

unknown_22:

unknown_23:???

unknown_24:	?

unknown_25:???

unknown_26:	?

unknown_27:
?#?

unknown_28:	?

unknown_29:
??

unknown_30:	?

unknown_31:??	?

unknown_32:	?

unknown_33:???

unknown_34:	?

unknown_35:
? ?

unknown_36:	?

unknown_37:
??

unknown_38:	?

unknown_39:???

unknown_40:	?

unknown_41:???

unknown_42:	?

unknown_43:
??

unknown_44:	?

unknown_45:
??

unknown_46:	?

unknown_47:
??

unknown_48:	?

unknown_49:
??

unknown_50:	?

unknown_51:	?(

unknown_52:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2284302
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?%
A__inference_model_layer_call_and_return_conditional_losses_229923

inputs=
#cone_conv2d_readvariableop_resource:02
$cone_biasadd_readvariableop_resource:0=
#conc_conv2d_readvariableop_resource:	$2
$conc_biasadd_readvariableop_resource:$=
#cona_conv2d_readvariableop_resource:2
$cona_biasadd_readvariableop_resource:>
$cone2_conv2d_readvariableop_resource:03
%cone2_biasadd_readvariableop_resource:>
$conc2_conv2d_readvariableop_resource:$3
%conc2_biasadd_readvariableop_resource:>
$cona2_conv2d_readvariableop_resource:3
%cona2_biasadd_readvariableop_resource:>
$cone3_conv2d_readvariableop_resource:3
%cone3_biasadd_readvariableop_resource:>
$conc3_conv2d_readvariableop_resource:3
%conc3_biasadd_readvariableop_resource:>
$cona3_conv2d_readvariableop_resource:3
%cona3_biasadd_readvariableop_resource:>
$cone4_conv2d_readvariableop_resource:3
%cone4_biasadd_readvariableop_resource:>
$conc4_conv2d_readvariableop_resource:3
%conc4_biasadd_readvariableop_resource:>
$cona4_conv2d_readvariableop_resource:3
%cona4_biasadd_readvariableop_resource:8
#hida_matmul_readvariableop_resource:???3
$hida_biasadd_readvariableop_resource:	?9
$hida2_matmul_readvariableop_resource:???4
%hida2_biasadd_readvariableop_resource:	?8
$hida3_matmul_readvariableop_resource:
?#?4
%hida3_biasadd_readvariableop_resource:	?8
$hida4_matmul_readvariableop_resource:
??4
%hida4_biasadd_readvariableop_resource:	?8
#hidc_matmul_readvariableop_resource:??	?3
$hidc_biasadd_readvariableop_resource:	?9
$hidc2_matmul_readvariableop_resource:???4
%hidc2_biasadd_readvariableop_resource:	?8
$hidc3_matmul_readvariableop_resource:
? ?4
%hidc3_biasadd_readvariableop_resource:	?8
$hidc4_matmul_readvariableop_resource:
??4
%hidc4_biasadd_readvariableop_resource:	?8
#hide_matmul_readvariableop_resource:???3
$hide_biasadd_readvariableop_resource:	?9
$hide2_matmul_readvariableop_resource:???4
%hide2_biasadd_readvariableop_resource:	?8
$hide3_matmul_readvariableop_resource:
??4
%hide3_biasadd_readvariableop_resource:	?8
$hide4_matmul_readvariableop_resource:
??4
%hide4_biasadd_readvariableop_resource:	?7
#hid1_matmul_readvariableop_resource:
??3
$hid1_biasadd_readvariableop_resource:	?7
#hid2_matmul_readvariableop_resource:
??3
$hid2_biasadd_readvariableop_resource:	?8
%output_matmul_readvariableop_resource:	?(4
&output_biasadd_readvariableop_resource:(
identity??ConA/BiasAdd/ReadVariableOp?ConA/Conv2D/ReadVariableOp?ConA2/BiasAdd/ReadVariableOp?ConA2/Conv2D/ReadVariableOp?ConA3/BiasAdd/ReadVariableOp?ConA3/Conv2D/ReadVariableOp?ConA4/BiasAdd/ReadVariableOp?ConA4/Conv2D/ReadVariableOp?ConC/BiasAdd/ReadVariableOp?ConC/Conv2D/ReadVariableOp?ConC2/BiasAdd/ReadVariableOp?ConC2/Conv2D/ReadVariableOp?ConC3/BiasAdd/ReadVariableOp?ConC3/Conv2D/ReadVariableOp?ConC4/BiasAdd/ReadVariableOp?ConC4/Conv2D/ReadVariableOp?ConE/BiasAdd/ReadVariableOp?ConE/Conv2D/ReadVariableOp?ConE2/BiasAdd/ReadVariableOp?ConE2/Conv2D/ReadVariableOp?ConE3/BiasAdd/ReadVariableOp?ConE3/Conv2D/ReadVariableOp?ConE4/BiasAdd/ReadVariableOp?ConE4/Conv2D/ReadVariableOp?Hid1/BiasAdd/ReadVariableOp?Hid1/MatMul/ReadVariableOp?Hid2/BiasAdd/ReadVariableOp?Hid2/MatMul/ReadVariableOp?HidA/BiasAdd/ReadVariableOp?HidA/MatMul/ReadVariableOp?HidA2/BiasAdd/ReadVariableOp?HidA2/MatMul/ReadVariableOp?HidA3/BiasAdd/ReadVariableOp?HidA3/MatMul/ReadVariableOp?HidA4/BiasAdd/ReadVariableOp?HidA4/MatMul/ReadVariableOp?HidC/BiasAdd/ReadVariableOp?HidC/MatMul/ReadVariableOp?HidC2/BiasAdd/ReadVariableOp?HidC2/MatMul/ReadVariableOp?HidC3/BiasAdd/ReadVariableOp?HidC3/MatMul/ReadVariableOp?HidC4/BiasAdd/ReadVariableOp?HidC4/MatMul/ReadVariableOp?HidE/BiasAdd/ReadVariableOp?HidE/MatMul/ReadVariableOp?HidE2/BiasAdd/ReadVariableOp?HidE2/MatMul/ReadVariableOp?HidE3/BiasAdd/ReadVariableOp?HidE3/MatMul/ReadVariableOp?HidE4/BiasAdd/ReadVariableOp?HidE4/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
ConE/Conv2D/ReadVariableOpReadVariableOp#cone_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02
ConE/Conv2D/ReadVariableOp?
ConE/Conv2DConv2Dinputs"ConE/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingVALID*
strides
2
ConE/Conv2D?
ConE/BiasAdd/ReadVariableOpReadVariableOp$cone_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
ConE/BiasAdd/ReadVariableOp?
ConE/BiasAddBiasAddConE/Conv2D:output:0#ConE/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02
ConE/BiasAddq
	ConE/ReluReluConE/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02
	ConE/Relu?
ConC/Conv2D/ReadVariableOpReadVariableOp#conc_conv2d_readvariableop_resource*&
_output_shapes
:	$*
dtype02
ConC/Conv2D/ReadVariableOp?
ConC/Conv2DConv2Dinputs"ConC/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$*
paddingVALID*
strides
2
ConC/Conv2D?
ConC/BiasAdd/ReadVariableOpReadVariableOp$conc_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
ConC/BiasAdd/ReadVariableOp?
ConC/BiasAddBiasAddConC/Conv2D:output:0#ConC/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$2
ConC/BiasAddq
	ConC/ReluReluConC/BiasAdd:output:0*
T0*1
_output_shapes
:???????????$2
	ConC/Relu?
ConA/Conv2D/ReadVariableOpReadVariableOp#cona_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConA/Conv2D/ReadVariableOp?
ConA/Conv2DConv2Dinputs"ConA/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
ConA/Conv2D?
ConA/BiasAdd/ReadVariableOpReadVariableOp$cona_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConA/BiasAdd/ReadVariableOp?
ConA/BiasAddBiasAddConA/Conv2D:output:0#ConA/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
ConA/BiasAddq
	ConA/ReluReluConA/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
	ConA/Relu?
PoolE/MaxPoolMaxPoolConE/Relu:activations:0*/
_output_shapes
:?????????B:0*
ksize
*
paddingVALID*
strides
2
PoolE/MaxPool?
PoolC/MaxPoolMaxPoolConC/Relu:activations:0*/
_output_shapes
:?????????F=$*
ksize
*
paddingVALID*
strides
2
PoolC/MaxPool?
PoolA/MaxPoolMaxPoolConA/Relu:activations:0*/
_output_shapes
:?????????I?*
ksize
*
paddingVALID*
strides
2
PoolA/MaxPoolw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_2/dropout/Const?
dropout_2/dropout/MulMulPoolE/MaxPool:output:0 dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:?????????B:02
dropout_2/dropout/Mulx
dropout_2/dropout/ShapeShapePoolE/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????B:0*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform?
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_2/dropout/GreaterEqual/y?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????B:02 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????B:02
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????B:02
dropout_2/dropout/Mul_1w
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_1/dropout/Const?
dropout_1/dropout/MulMulPoolC/MaxPool:output:0 dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:?????????F=$2
dropout_1/dropout/Mulx
dropout_1/dropout/ShapeShapePoolC/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????F=$*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????F=$2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????F=$2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????F=$2
dropout_1/dropout/Mul_1s
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/dropout/Const?
dropout/dropout/MulMulPoolA/MaxPool:output:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:?????????I?2
dropout/dropout/Mult
dropout/dropout/ShapeShapePoolA/MaxPool:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????I?*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????I?2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????I?2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????I?2
dropout/dropout/Mul_1?
ConE2/Conv2D/ReadVariableOpReadVariableOp$cone2_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02
ConE2/Conv2D/ReadVariableOp?
ConE2/Conv2DConv2Ddropout_2/dropout/Mul_1:z:0#ConE2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@8*
paddingVALID*
strides
2
ConE2/Conv2D?
ConE2/BiasAdd/ReadVariableOpReadVariableOp%cone2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConE2/BiasAdd/ReadVariableOp?
ConE2/BiasAddBiasAddConE2/Conv2D:output:0$ConE2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@82
ConE2/BiasAddr

ConE2/ReluReluConE2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@82

ConE2/Relu?
ConC2/Conv2D/ReadVariableOpReadVariableOp$conc2_conv2d_readvariableop_resource*&
_output_shapes
:$*
dtype02
ConC2/Conv2D/ReadVariableOp?
ConC2/Conv2DConv2Ddropout_1/dropout/Mul_1:z:0#ConC2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????D;*
paddingVALID*
strides
2
ConC2/Conv2D?
ConC2/BiasAdd/ReadVariableOpReadVariableOp%conc2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConC2/BiasAdd/ReadVariableOp?
ConC2/BiasAddBiasAddConC2/Conv2D:output:0$ConC2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????D;2
ConC2/BiasAddr

ConC2/ReluReluConC2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????D;2

ConC2/Relu?
ConA2/Conv2D/ReadVariableOpReadVariableOp$cona2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConA2/Conv2D/ReadVariableOp?
ConA2/Conv2DConv2Ddropout/dropout/Mul_1:z:0#ConA2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????G=*
paddingVALID*
strides
2
ConA2/Conv2D?
ConA2/BiasAdd/ReadVariableOpReadVariableOp%cona2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConA2/BiasAdd/ReadVariableOp?
ConA2/BiasAddBiasAddConA2/Conv2D:output:0$ConA2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????G=2
ConA2/BiasAddr

ConA2/ReluReluConA2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????G=2

ConA2/Relu?
PoolE2/MaxPoolMaxPoolConE2/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
PoolE2/MaxPool?
PoolC2/MaxPoolMaxPoolConC2/Relu:activations:0*/
_output_shapes
:?????????"*
ksize
*
paddingVALID*
strides
2
PoolC2/MaxPool?
PoolA2/MaxPoolMaxPoolConA2/Relu:activations:0*/
_output_shapes
:?????????#*
ksize
*
paddingVALID*
strides
2
PoolA2/MaxPool?
ConE3/Conv2D/ReadVariableOpReadVariableOp$cone3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConE3/Conv2D/ReadVariableOp?
ConE3/Conv2DConv2DPoolE2/MaxPool:output:0#ConE3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
ConE3/Conv2D?
ConE3/BiasAdd/ReadVariableOpReadVariableOp%cone3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConE3/BiasAdd/ReadVariableOp?
ConE3/BiasAddBiasAddConE3/Conv2D:output:0$ConE3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
ConE3/BiasAddr

ConE3/ReluReluConE3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2

ConE3/Relu?
ConC3/Conv2D/ReadVariableOpReadVariableOp$conc3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConC3/Conv2D/ReadVariableOp?
ConC3/Conv2DConv2DPoolC2/MaxPool:output:0#ConC3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
ConC3/Conv2D?
ConC3/BiasAdd/ReadVariableOpReadVariableOp%conc3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConC3/BiasAdd/ReadVariableOp?
ConC3/BiasAddBiasAddConC3/Conv2D:output:0$ConC3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
ConC3/BiasAddr

ConC3/ReluReluConC3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2

ConC3/Relu?
ConA3/Conv2D/ReadVariableOpReadVariableOp$cona3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConA3/Conv2D/ReadVariableOp?
ConA3/Conv2DConv2DPoolA2/MaxPool:output:0#ConA3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!*
paddingVALID*
strides
2
ConA3/Conv2D?
ConA3/BiasAdd/ReadVariableOpReadVariableOp%cona3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConA3/BiasAdd/ReadVariableOp?
ConA3/BiasAddBiasAddConA3/Conv2D:output:0$ConA3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!2
ConA3/BiasAddr

ConA3/ReluReluConA3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????!2

ConA3/Relu?
PoolE3/MaxPoolMaxPoolConE3/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
PoolE3/MaxPool?
PoolC3/MaxPoolMaxPoolConC3/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
PoolC3/MaxPool?
PoolA3/MaxPoolMaxPoolConA3/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
PoolA3/MaxPool?
ConE4/Conv2D/ReadVariableOpReadVariableOp$cone4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConE4/Conv2D/ReadVariableOp?
ConE4/Conv2DConv2DPoolE3/MaxPool:output:0#ConE4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
ConE4/Conv2D?
ConE4/BiasAdd/ReadVariableOpReadVariableOp%cone4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConE4/BiasAdd/ReadVariableOp?
ConE4/BiasAddBiasAddConE4/Conv2D:output:0$ConE4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
ConE4/BiasAddr

ConE4/ReluReluConE4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2

ConE4/Relu?
ConC4/Conv2D/ReadVariableOpReadVariableOp$conc4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConC4/Conv2D/ReadVariableOp?
ConC4/Conv2DConv2DPoolC3/MaxPool:output:0#ConC4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
ConC4/Conv2D?
ConC4/BiasAdd/ReadVariableOpReadVariableOp%conc4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConC4/BiasAdd/ReadVariableOp?
ConC4/BiasAddBiasAddConC4/Conv2D:output:0$ConC4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
ConC4/BiasAddr

ConC4/ReluReluConC4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2

ConC4/Relu?
ConA4/Conv2D/ReadVariableOpReadVariableOp$cona4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConA4/Conv2D/ReadVariableOp?
ConA4/Conv2DConv2DPoolA3/MaxPool:output:0#ConA4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
ConA4/Conv2D?
ConA4/BiasAdd/ReadVariableOpReadVariableOp%cona4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConA4/BiasAdd/ReadVariableOp?
ConA4/BiasAddBiasAddConA4/Conv2D:output:0$ConA4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
ConA4/BiasAddr

ConA4/ReluReluConA4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2

ConA4/Relu?
PoolE4/MaxPoolMaxPoolConE4/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
PoolE4/MaxPool?
PoolC4/MaxPoolMaxPoolConC4/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
PoolC4/MaxPool?
PoolA4/MaxPoolMaxPoolConA4/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
PoolA4/MaxPooly
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_11/dropout/Const?
dropout_11/dropout/MulMulPoolE4/MaxPool:output:0!dropout_11/dropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout_11/dropout/Mul{
dropout_11/dropout/ShapeShapePoolE4/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_11/dropout/Shape?
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype021
/dropout_11/dropout/random_uniform/RandomUniform?
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_11/dropout/GreaterEqual/y?
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2!
dropout_11/dropout/GreaterEqual?
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout_11/dropout/Cast?
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout_11/dropout/Mul_1y
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_10/dropout/Const?
dropout_10/dropout/MulMulPoolE3/MaxPool:output:0!dropout_10/dropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout_10/dropout/Mul{
dropout_10/dropout/ShapeShapePoolE3/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_10/dropout/Shape?
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype021
/dropout_10/dropout/random_uniform/RandomUniform?
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_10/dropout/GreaterEqual/y?
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2!
dropout_10/dropout/GreaterEqual?
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout_10/dropout/Cast?
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout_10/dropout/Mul_1w
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_9/dropout/Const?
dropout_9/dropout/MulMulPoolE2/MaxPool:output:0 dropout_9/dropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout_9/dropout/Muly
dropout_9/dropout/ShapeShapePoolE2/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shape?
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype020
.dropout_9/dropout/random_uniform/RandomUniform?
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_9/dropout/GreaterEqual/y?
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2 
dropout_9/dropout/GreaterEqual?
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout_9/dropout/Cast?
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout_9/dropout/Mul_1w
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_8/dropout/Const?
dropout_8/dropout/MulMulPoolC4/MaxPool:output:0 dropout_8/dropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout_8/dropout/Muly
dropout_8/dropout/ShapeShapePoolC4/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_8/dropout/Shape?
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype020
.dropout_8/dropout/random_uniform/RandomUniform?
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_8/dropout/GreaterEqual/y?
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2 
dropout_8/dropout/GreaterEqual?
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout_8/dropout/Cast?
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout_8/dropout/Mul_1w
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_7/dropout/Const?
dropout_7/dropout/MulMulPoolC3/MaxPool:output:0 dropout_7/dropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout_7/dropout/Muly
dropout_7/dropout/ShapeShapePoolC3/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_7/dropout/Shape?
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype020
.dropout_7/dropout/random_uniform/RandomUniform?
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_7/dropout/GreaterEqual/y?
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2 
dropout_7/dropout/GreaterEqual?
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout_7/dropout/Cast?
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout_7/dropout/Mul_1w
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_6/dropout/Const?
dropout_6/dropout/MulMulPoolC2/MaxPool:output:0 dropout_6/dropout/Const:output:0*
T0*/
_output_shapes
:?????????"2
dropout_6/dropout/Muly
dropout_6/dropout/ShapeShapePoolC2/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_6/dropout/Shape?
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????"*
dtype020
.dropout_6/dropout/random_uniform/RandomUniform?
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_6/dropout/GreaterEqual/y?
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????"2 
dropout_6/dropout/GreaterEqual?
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????"2
dropout_6/dropout/Cast?
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????"2
dropout_6/dropout/Mul_1w
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_5/dropout/Const?
dropout_5/dropout/MulMulPoolA4/MaxPool:output:0 dropout_5/dropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout_5/dropout/Muly
dropout_5/dropout/ShapeShapePoolA4/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_5/dropout/Shape?
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype020
.dropout_5/dropout/random_uniform/RandomUniform?
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_5/dropout/GreaterEqual/y?
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2 
dropout_5/dropout/GreaterEqual?
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout_5/dropout/Cast?
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout_5/dropout/Mul_1w
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_4/dropout/Const?
dropout_4/dropout/MulMulPoolA3/MaxPool:output:0 dropout_4/dropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout_4/dropout/Muly
dropout_4/dropout/ShapeShapePoolA3/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_4/dropout/Shape?
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype020
.dropout_4/dropout/random_uniform/RandomUniform?
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_4/dropout/GreaterEqual/y?
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2 
dropout_4/dropout/GreaterEqual?
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout_4/dropout/Cast?
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout_4/dropout/Mul_1w
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_3/dropout/Const?
dropout_3/dropout/MulMulPoolA2/MaxPool:output:0 dropout_3/dropout/Const:output:0*
T0*/
_output_shapes
:?????????#2
dropout_3/dropout/Muly
dropout_3/dropout/ShapeShapePoolA2/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shape?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????#*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform?
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_3/dropout/GreaterEqual/y?
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????#2 
dropout_3/dropout/GreaterEqual?
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????#2
dropout_3/dropout/Cast?
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????#2
dropout_3/dropout/Mul_1u
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"????X  2
flatten_11/Const?
flatten_11/ReshapeReshapedropout_11/dropout/Mul_1:z:0flatten_11/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_11/Reshapeu
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"????<  2
flatten_10/Const?
flatten_10/ReshapeReshapedropout_10/dropout/Mul_1:z:0flatten_10/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_10/Reshapes
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? F  2
flatten_9/Const?
flatten_9/ReshapeReshapedropout_9/dropout/Mul_1:z:0flatten_9/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_9/Reshapes
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????? 2
flatten_8/Const?
flatten_8/ReshapeReshapePoolE/MaxPool:output:0flatten_8/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_8/Reshapes
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_7/Const?
flatten_7/ReshapeReshapedropout_8/dropout/Mul_1:z:0flatten_7/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_7/Reshapes
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
flatten_6/Const?
flatten_6/ReshapeReshapedropout_7/dropout/Mul_1:z:0flatten_6/Const:output:0*
T0*(
_output_shapes
:?????????? 2
flatten_6/Reshapes
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????M  2
flatten_5/Const?
flatten_5/ReshapeReshapedropout_6/dropout/Mul_1:z:0flatten_5/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_5/Reshapes
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????xX 2
flatten_4/Const?
flatten_4/ReshapeReshapePoolC/MaxPool:output:0flatten_4/Const:output:0*
T0*)
_output_shapes
:???????????	2
flatten_4/Reshapes
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????H  2
flatten_3/Const?
flatten_3/ReshapeReshapedropout_5/dropout/Mul_1:z:0flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_3/Reshapes
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_2/Const?
flatten_2/ReshapeReshapedropout_4/dropout/Mul_1:z:0flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????#2
flatten_2/Reshapes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????R  2
flatten_1/Const?
flatten_1/ReshapeReshapedropout_3/dropout/Mul_1:z:0flatten_1/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_1/Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????(? 2
flatten/Const?
flatten/ReshapeReshapePoolA/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten/Reshape?
HidA/MatMul/ReadVariableOpReadVariableOp#hida_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
HidA/MatMul/ReadVariableOp?
HidA/MatMulMatMulflatten/Reshape:output:0"HidA/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA/MatMul?
HidA/BiasAdd/ReadVariableOpReadVariableOp$hida_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidA/BiasAdd/ReadVariableOp?
HidA/BiasAddBiasAddHidA/MatMul:product:0#HidA/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA/BiasAddh
	HidA/ReluReluHidA/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
	HidA/Relu?
HidA2/MatMul/ReadVariableOpReadVariableOp$hida2_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
HidA2/MatMul/ReadVariableOp?
HidA2/MatMulMatMulflatten_1/Reshape:output:0#HidA2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA2/MatMul?
HidA2/BiasAdd/ReadVariableOpReadVariableOp%hida2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidA2/BiasAdd/ReadVariableOp?
HidA2/BiasAddBiasAddHidA2/MatMul:product:0$HidA2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA2/BiasAddk

HidA2/ReluReluHidA2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidA2/Relu?
HidA3/MatMul/ReadVariableOpReadVariableOp$hida3_matmul_readvariableop_resource* 
_output_shapes
:
?#?*
dtype02
HidA3/MatMul/ReadVariableOp?
HidA3/MatMulMatMulflatten_2/Reshape:output:0#HidA3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA3/MatMul?
HidA3/BiasAdd/ReadVariableOpReadVariableOp%hida3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidA3/BiasAdd/ReadVariableOp?
HidA3/BiasAddBiasAddHidA3/MatMul:product:0$HidA3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA3/BiasAddk

HidA3/ReluReluHidA3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidA3/Relu?
HidA4/MatMul/ReadVariableOpReadVariableOp$hida4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
HidA4/MatMul/ReadVariableOp?
HidA4/MatMulMatMulflatten_3/Reshape:output:0#HidA4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA4/MatMul?
HidA4/BiasAdd/ReadVariableOpReadVariableOp%hida4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidA4/BiasAdd/ReadVariableOp?
HidA4/BiasAddBiasAddHidA4/MatMul:product:0$HidA4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA4/BiasAddk

HidA4/ReluReluHidA4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidA4/Relu?
HidC/MatMul/ReadVariableOpReadVariableOp#hidc_matmul_readvariableop_resource*!
_output_shapes
:??	?*
dtype02
HidC/MatMul/ReadVariableOp?
HidC/MatMulMatMulflatten_4/Reshape:output:0"HidC/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidC/MatMul?
HidC/BiasAdd/ReadVariableOpReadVariableOp$hidc_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidC/BiasAdd/ReadVariableOp?
HidC/BiasAddBiasAddHidC/MatMul:product:0#HidC/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidC/BiasAddh
	HidC/ReluReluHidC/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
	HidC/Relu?
HidC2/MatMul/ReadVariableOpReadVariableOp$hidc2_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
HidC2/MatMul/ReadVariableOp?
HidC2/MatMulMatMulflatten_5/Reshape:output:0#HidC2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidC2/MatMul?
HidC2/BiasAdd/ReadVariableOpReadVariableOp%hidc2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidC2/BiasAdd/ReadVariableOp?
HidC2/BiasAddBiasAddHidC2/MatMul:product:0$HidC2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidC2/BiasAddk

HidC2/ReluReluHidC2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidC2/Relu?
HidC3/MatMul/ReadVariableOpReadVariableOp$hidc3_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02
HidC3/MatMul/ReadVariableOp?
HidC3/MatMulMatMulflatten_6/Reshape:output:0#HidC3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidC3/MatMul?
HidC3/BiasAdd/ReadVariableOpReadVariableOp%hidc3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidC3/BiasAdd/ReadVariableOp?
HidC3/BiasAddBiasAddHidC3/MatMul:product:0$HidC3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidC3/BiasAddk

HidC3/ReluReluHidC3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidC3/Relu?
HidC4/MatMul/ReadVariableOpReadVariableOp$hidc4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
HidC4/MatMul/ReadVariableOp?
HidC4/MatMulMatMulflatten_7/Reshape:output:0#HidC4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidC4/MatMul?
HidC4/BiasAdd/ReadVariableOpReadVariableOp%hidc4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidC4/BiasAdd/ReadVariableOp?
HidC4/BiasAddBiasAddHidC4/MatMul:product:0$HidC4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidC4/BiasAddk

HidC4/ReluReluHidC4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidC4/Relu?
HidE/MatMul/ReadVariableOpReadVariableOp#hide_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
HidE/MatMul/ReadVariableOp?
HidE/MatMulMatMulflatten_8/Reshape:output:0"HidE/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidE/MatMul?
HidE/BiasAdd/ReadVariableOpReadVariableOp$hide_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidE/BiasAdd/ReadVariableOp?
HidE/BiasAddBiasAddHidE/MatMul:product:0#HidE/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidE/BiasAddh
	HidE/ReluReluHidE/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
	HidE/Relu?
HidE2/MatMul/ReadVariableOpReadVariableOp$hide2_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
HidE2/MatMul/ReadVariableOp?
HidE2/MatMulMatMulflatten_9/Reshape:output:0#HidE2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidE2/MatMul?
HidE2/BiasAdd/ReadVariableOpReadVariableOp%hide2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidE2/BiasAdd/ReadVariableOp?
HidE2/BiasAddBiasAddHidE2/MatMul:product:0$HidE2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidE2/BiasAddk

HidE2/ReluReluHidE2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidE2/Relu?
HidE3/MatMul/ReadVariableOpReadVariableOp$hide3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
HidE3/MatMul/ReadVariableOp?
HidE3/MatMulMatMulflatten_10/Reshape:output:0#HidE3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidE3/MatMul?
HidE3/BiasAdd/ReadVariableOpReadVariableOp%hide3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidE3/BiasAdd/ReadVariableOp?
HidE3/BiasAddBiasAddHidE3/MatMul:product:0$HidE3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidE3/BiasAddk

HidE3/ReluReluHidE3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidE3/Relu?
HidE4/MatMul/ReadVariableOpReadVariableOp$hide4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
HidE4/MatMul/ReadVariableOp?
HidE4/MatMulMatMulflatten_11/Reshape:output:0#HidE4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidE4/MatMul?
HidE4/BiasAdd/ReadVariableOpReadVariableOp%hide4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidE4/BiasAdd/ReadVariableOp?
HidE4/BiasAddBiasAddHidE4/MatMul:product:0$HidE4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidE4/BiasAddk

HidE4/ReluReluHidE4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidE4/Relut
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2HidA/Relu:activations:0HidA2/Relu:activations:0HidA3/Relu:activations:0HidA4/Relu:activations:0HidC/Relu:activations:0HidC2/Relu:activations:0HidC3/Relu:activations:0HidC4/Relu:activations:0HidE/Relu:activations:0HidE2/Relu:activations:0HidE3/Relu:activations:0HidE4/Relu:activations:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate/concat?
Hid1/MatMul/ReadVariableOpReadVariableOp#hid1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Hid1/MatMul/ReadVariableOp?
Hid1/MatMulMatMulconcatenate/concat:output:0"Hid1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Hid1/MatMul?
Hid1/BiasAdd/ReadVariableOpReadVariableOp$hid1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
Hid1/BiasAdd/ReadVariableOp?
Hid1/BiasAddBiasAddHid1/MatMul:product:0#Hid1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Hid1/BiasAddh
	Hid1/ReluReluHid1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
	Hid1/Relu?
Hid2/MatMul/ReadVariableOpReadVariableOp#hid2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Hid2/MatMul/ReadVariableOp?
Hid2/MatMulMatMulHid1/Relu:activations:0"Hid2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Hid2/MatMul?
Hid2/BiasAdd/ReadVariableOpReadVariableOp$hid2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
Hid2/BiasAdd/ReadVariableOp?
Hid2/BiasAddBiasAddHid2/MatMul:product:0#Hid2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Hid2/BiasAddh
	Hid2/TanhTanhHid2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
	Hid2/Tanh?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	?(*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMulHid2/Tanh:y:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
output/Softmaxs
IdentityIdentityoutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????(2

Identity?
NoOpNoOp^ConA/BiasAdd/ReadVariableOp^ConA/Conv2D/ReadVariableOp^ConA2/BiasAdd/ReadVariableOp^ConA2/Conv2D/ReadVariableOp^ConA3/BiasAdd/ReadVariableOp^ConA3/Conv2D/ReadVariableOp^ConA4/BiasAdd/ReadVariableOp^ConA4/Conv2D/ReadVariableOp^ConC/BiasAdd/ReadVariableOp^ConC/Conv2D/ReadVariableOp^ConC2/BiasAdd/ReadVariableOp^ConC2/Conv2D/ReadVariableOp^ConC3/BiasAdd/ReadVariableOp^ConC3/Conv2D/ReadVariableOp^ConC4/BiasAdd/ReadVariableOp^ConC4/Conv2D/ReadVariableOp^ConE/BiasAdd/ReadVariableOp^ConE/Conv2D/ReadVariableOp^ConE2/BiasAdd/ReadVariableOp^ConE2/Conv2D/ReadVariableOp^ConE3/BiasAdd/ReadVariableOp^ConE3/Conv2D/ReadVariableOp^ConE4/BiasAdd/ReadVariableOp^ConE4/Conv2D/ReadVariableOp^Hid1/BiasAdd/ReadVariableOp^Hid1/MatMul/ReadVariableOp^Hid2/BiasAdd/ReadVariableOp^Hid2/MatMul/ReadVariableOp^HidA/BiasAdd/ReadVariableOp^HidA/MatMul/ReadVariableOp^HidA2/BiasAdd/ReadVariableOp^HidA2/MatMul/ReadVariableOp^HidA3/BiasAdd/ReadVariableOp^HidA3/MatMul/ReadVariableOp^HidA4/BiasAdd/ReadVariableOp^HidA4/MatMul/ReadVariableOp^HidC/BiasAdd/ReadVariableOp^HidC/MatMul/ReadVariableOp^HidC2/BiasAdd/ReadVariableOp^HidC2/MatMul/ReadVariableOp^HidC3/BiasAdd/ReadVariableOp^HidC3/MatMul/ReadVariableOp^HidC4/BiasAdd/ReadVariableOp^HidC4/MatMul/ReadVariableOp^HidE/BiasAdd/ReadVariableOp^HidE/MatMul/ReadVariableOp^HidE2/BiasAdd/ReadVariableOp^HidE2/MatMul/ReadVariableOp^HidE3/BiasAdd/ReadVariableOp^HidE3/MatMul/ReadVariableOp^HidE4/BiasAdd/ReadVariableOp^HidE4/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
ConA/BiasAdd/ReadVariableOpConA/BiasAdd/ReadVariableOp28
ConA/Conv2D/ReadVariableOpConA/Conv2D/ReadVariableOp2<
ConA2/BiasAdd/ReadVariableOpConA2/BiasAdd/ReadVariableOp2:
ConA2/Conv2D/ReadVariableOpConA2/Conv2D/ReadVariableOp2<
ConA3/BiasAdd/ReadVariableOpConA3/BiasAdd/ReadVariableOp2:
ConA3/Conv2D/ReadVariableOpConA3/Conv2D/ReadVariableOp2<
ConA4/BiasAdd/ReadVariableOpConA4/BiasAdd/ReadVariableOp2:
ConA4/Conv2D/ReadVariableOpConA4/Conv2D/ReadVariableOp2:
ConC/BiasAdd/ReadVariableOpConC/BiasAdd/ReadVariableOp28
ConC/Conv2D/ReadVariableOpConC/Conv2D/ReadVariableOp2<
ConC2/BiasAdd/ReadVariableOpConC2/BiasAdd/ReadVariableOp2:
ConC2/Conv2D/ReadVariableOpConC2/Conv2D/ReadVariableOp2<
ConC3/BiasAdd/ReadVariableOpConC3/BiasAdd/ReadVariableOp2:
ConC3/Conv2D/ReadVariableOpConC3/Conv2D/ReadVariableOp2<
ConC4/BiasAdd/ReadVariableOpConC4/BiasAdd/ReadVariableOp2:
ConC4/Conv2D/ReadVariableOpConC4/Conv2D/ReadVariableOp2:
ConE/BiasAdd/ReadVariableOpConE/BiasAdd/ReadVariableOp28
ConE/Conv2D/ReadVariableOpConE/Conv2D/ReadVariableOp2<
ConE2/BiasAdd/ReadVariableOpConE2/BiasAdd/ReadVariableOp2:
ConE2/Conv2D/ReadVariableOpConE2/Conv2D/ReadVariableOp2<
ConE3/BiasAdd/ReadVariableOpConE3/BiasAdd/ReadVariableOp2:
ConE3/Conv2D/ReadVariableOpConE3/Conv2D/ReadVariableOp2<
ConE4/BiasAdd/ReadVariableOpConE4/BiasAdd/ReadVariableOp2:
ConE4/Conv2D/ReadVariableOpConE4/Conv2D/ReadVariableOp2:
Hid1/BiasAdd/ReadVariableOpHid1/BiasAdd/ReadVariableOp28
Hid1/MatMul/ReadVariableOpHid1/MatMul/ReadVariableOp2:
Hid2/BiasAdd/ReadVariableOpHid2/BiasAdd/ReadVariableOp28
Hid2/MatMul/ReadVariableOpHid2/MatMul/ReadVariableOp2:
HidA/BiasAdd/ReadVariableOpHidA/BiasAdd/ReadVariableOp28
HidA/MatMul/ReadVariableOpHidA/MatMul/ReadVariableOp2<
HidA2/BiasAdd/ReadVariableOpHidA2/BiasAdd/ReadVariableOp2:
HidA2/MatMul/ReadVariableOpHidA2/MatMul/ReadVariableOp2<
HidA3/BiasAdd/ReadVariableOpHidA3/BiasAdd/ReadVariableOp2:
HidA3/MatMul/ReadVariableOpHidA3/MatMul/ReadVariableOp2<
HidA4/BiasAdd/ReadVariableOpHidA4/BiasAdd/ReadVariableOp2:
HidA4/MatMul/ReadVariableOpHidA4/MatMul/ReadVariableOp2:
HidC/BiasAdd/ReadVariableOpHidC/BiasAdd/ReadVariableOp28
HidC/MatMul/ReadVariableOpHidC/MatMul/ReadVariableOp2<
HidC2/BiasAdd/ReadVariableOpHidC2/BiasAdd/ReadVariableOp2:
HidC2/MatMul/ReadVariableOpHidC2/MatMul/ReadVariableOp2<
HidC3/BiasAdd/ReadVariableOpHidC3/BiasAdd/ReadVariableOp2:
HidC3/MatMul/ReadVariableOpHidC3/MatMul/ReadVariableOp2<
HidC4/BiasAdd/ReadVariableOpHidC4/BiasAdd/ReadVariableOp2:
HidC4/MatMul/ReadVariableOpHidC4/MatMul/ReadVariableOp2:
HidE/BiasAdd/ReadVariableOpHidE/BiasAdd/ReadVariableOp28
HidE/MatMul/ReadVariableOpHidE/MatMul/ReadVariableOp2<
HidE2/BiasAdd/ReadVariableOpHidE2/BiasAdd/ReadVariableOp2:
HidE2/MatMul/ReadVariableOpHidE2/MatMul/ReadVariableOp2<
HidE3/BiasAdd/ReadVariableOpHidE3/BiasAdd/ReadVariableOp2:
HidE3/MatMul/ReadVariableOpHidE3/MatMul/ReadVariableOp2<
HidE4/BiasAdd/ReadVariableOpHidE4/BiasAdd/ReadVariableOp2:
HidE4/MatMul/ReadVariableOpHidE4/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
A__inference_HidE2_layer_call_and_return_conditional_losses_227222

inputs3
matmul_readvariableop_resource:???.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
e
F__inference_dropout_10_layer_call_and_return_conditional_losses_227864

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_PoolA4_layer_call_and_return_conditional_losses_230444

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_5_layer_call_fn_230787

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_2270162
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????":W S
/
_output_shapes
:?????????"
 
_user_specified_nameinputs
?
]
A__inference_PoolC_layer_call_and_return_conditional_losses_226362

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_HidE3_layer_call_and_return_conditional_losses_227239

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_230085

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????F=$2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????F=$2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????F=$:W S
/
_output_shapes
:?????????F=$
 
_user_specified_nameinputs
?
^
B__inference_PoolE4_layer_call_and_return_conditional_losses_230479

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
^
B__inference_PoolE4_layer_call_and_return_conditional_losses_230484

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
B
&__inference_PoolA_layer_call_fn_229988

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
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolA_layer_call_and_return_conditional_losses_2263402
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_226676

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????B:02

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????B:02

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????B:0:W S
/
_output_shapes
:?????????B:0
 
_user_specified_nameinputs
?
F
*__inference_dropout_1_layer_call_fn_230075

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
:?????????F=$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_2266832
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????F=$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????F=$:W S
/
_output_shapes
:?????????F=$
 
_user_specified_nameinputs
?
d
E__inference_dropout_8_layer_call_and_return_conditional_losses_230646

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
A__inference_ConC2_layer_call_and_return_conditional_losses_226720

inputs8
conv2d_readvariableop_resource:$-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????D;*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????D;2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????D;2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????D;2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????F=$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????F=$
 
_user_specified_nameinputs
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_230058

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????I?2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????I?2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????I?:W S
/
_output_shapes
:?????????I?
 
_user_specified_nameinputs
?
?
%__inference_Hid1_layer_call_fn_231141

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_Hid1_layer_call_and_return_conditional_losses_2272922
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
^
B__inference_PoolE4_layer_call_and_return_conditional_losses_226582

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
^
B__inference_PoolE3_layer_call_and_return_conditional_losses_230359

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_9_layer_call_and_return_conditional_losses_230673

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
C
'__inference_PoolA4_layer_call_fn_230429

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
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA4_layer_call_and_return_conditional_losses_2265382
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_8_layer_call_and_return_conditional_losses_230826

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????? 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????B:0:W S
/
_output_shapes
:?????????B:0
 
_user_specified_nameinputs
?
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_226960

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????#2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????#2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????#:W S
/
_output_shapes
:?????????#
 
_user_specified_nameinputs
?
b
C__inference_dropout_layer_call_and_return_conditional_losses_230070

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????I?2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????I?*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????I?2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????I?2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????I?2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????I?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????I?:W S
/
_output_shapes
:?????????I?
 
_user_specified_nameinputs
?
C
'__inference_PoolE4_layer_call_fn_230469

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
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolE4_layer_call_and_return_conditional_losses_2265822
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_227048

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????R  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????#:W S
/
_output_shapes
:?????????#
 
_user_specified_nameinputs
?
B
&__inference_PoolE_layer_call_fn_230033

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
:?????????B:0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolE_layer_call_and_return_conditional_losses_2266572
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????B:02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????0:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?
C
'__inference_PoolC3_layer_call_fn_230334

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolC3_layer_call_and_return_conditional_losses_2268222
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_230124

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????B:02
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????B:0*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????B:02
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????B:02
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????B:02
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????B:02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????B:0:W S
/
_output_shapes
:?????????B:0
 
_user_specified_nameinputs
?
a
E__inference_flatten_8_layer_call_and_return_conditional_losses_226992

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????? 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????B:0:W S
/
_output_shapes
:?????????B:0
 
_user_specified_nameinputs
?
?
B__inference_output_layer_call_and_return_conditional_losses_227326

inputs1
matmul_readvariableop_resource:	?(-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????(2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????(2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_7_layer_call_and_return_conditional_losses_227795

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_ConC4_layer_call_fn_230393

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConC4_layer_call_and_return_conditional_losses_2268582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
]
A__inference_PoolA_layer_call_and_return_conditional_losses_230003

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????I?*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????I?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_5_layer_call_and_return_conditional_losses_230793

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????M  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????":W S
/
_output_shapes
:?????????"
 
_user_specified_nameinputs
?
C
'__inference_PoolA4_layer_call_fn_230434

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA4_layer_call_and_return_conditional_losses_2268972
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
A__inference_HidA4_layer_call_and_return_conditional_losses_230939

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_4_layer_call_and_return_conditional_losses_230782

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????xX 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????	2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????F=$:W S
/
_output_shapes
:?????????F=$
 
_user_specified_nameinputs
?
B
&__inference_PoolA_layer_call_fn_229993

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
:?????????I?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolA_layer_call_and_return_conditional_losses_2266692
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????I?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
A__inference_HidA4_layer_call_and_return_conditional_losses_227120

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
^
B__inference_PoolE4_layer_call_and_return_conditional_losses_226885

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
A__inference_ConC4_layer_call_and_return_conditional_losses_230404

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_concatenate_layer_call_fn_231115
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_2272792
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/2:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/3:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/4:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/5:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/6:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/7:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/8:R	N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/9:S
O
(
_output_shapes
:??????????
#
_user_specified_name	inputs/10:SO
(
_output_shapes
:??????????
#
_user_specified_name	inputs/11
?
^
B__inference_PoolA3_layer_call_and_return_conditional_losses_230319

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_HidE3_layer_call_and_return_conditional_losses_231079

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
A__inference_ConA4_layer_call_and_return_conditional_losses_226875

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
C
'__inference_PoolE4_layer_call_fn_230474

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolE4_layer_call_and_return_conditional_losses_2268852
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_11_layer_call_and_return_conditional_losses_230715

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
F__inference_dropout_10_layer_call_and_return_conditional_losses_230700

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_concatenate_layer_call_and_return_conditional_losses_231132
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/2:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/3:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/4:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/5:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/6:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/7:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/8:R	N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/9:S
O
(
_output_shapes
:??????????
#
_user_specified_name	inputs/10:SO
(
_output_shapes
:??????????
#
_user_specified_name	inputs/11
?
?
&__inference_model_layer_call_fn_229240

inputs!
unknown:0
	unknown_0:0#
	unknown_1:	$
	unknown_2:$#
	unknown_3:
	unknown_4:#
	unknown_5:0
	unknown_6:#
	unknown_7:$
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:$

unknown_15:

unknown_16:$

unknown_17:

unknown_18:$

unknown_19:

unknown_20:$

unknown_21:

unknown_22:

unknown_23:???

unknown_24:	?

unknown_25:???

unknown_26:	?

unknown_27:
?#?

unknown_28:	?

unknown_29:
??

unknown_30:	?

unknown_31:??	?

unknown_32:	?

unknown_33:???

unknown_34:	?

unknown_35:
? ?

unknown_36:	?

unknown_37:
??

unknown_38:	?

unknown_39:???

unknown_40:	?

unknown_41:???

unknown_42:	?

unknown_43:
??

unknown_44:	?

unknown_45:
??

unknown_46:	?

unknown_47:
??

unknown_48:	?

unknown_49:
??

unknown_50:	?

unknown_51:	?(

unknown_52:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2273332
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
A__inference_HidA3_layer_call_and_return_conditional_losses_227103

inputs2
matmul_readvariableop_resource:
?#?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?#?*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????#
 
_user_specified_nameinputs
?
c
*__inference_dropout_2_layer_call_fn_230107

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????B:0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2280912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????B:02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????B:022
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????B:0
 
_user_specified_nameinputs
?
?
A__inference_HidE2_layer_call_and_return_conditional_losses_231059

inputs3
matmul_readvariableop_resource:???.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
@__inference_ConA_layer_call_and_return_conditional_losses_229943

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_6_layer_call_fn_230570

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
:?????????"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_2269392
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????":W S
/
_output_shapes
:?????????"
 
_user_specified_nameinputs
?
b
F__inference_flatten_11_layer_call_and_return_conditional_losses_226968

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????X  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
A__inference_ConE2_layer_call_and_return_conditional_losses_226703

inputs8
conv2d_readvariableop_resource:0-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@8*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@82	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@82
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@82

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????B:0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????B:0
 
_user_specified_nameinputs
?
C
'__inference_PoolE2_layer_call_fn_230229

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
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolE2_layer_call_and_return_conditional_losses_2264502
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
^
B__inference_PoolC2_layer_call_and_return_conditional_losses_226428

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_6_layer_call_and_return_conditional_losses_230804

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????? 2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_HidE2_layer_call_fn_231048

inputs
unknown:???
	unknown_0:	?
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
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidE2_layer_call_and_return_conditional_losses_2272222
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
@__inference_Hid2_layer_call_and_return_conditional_losses_231172

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Tanhd
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
@__inference_ConE_layer_call_and_return_conditional_losses_226613

inputs8
conv2d_readvariableop_resource:0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????02
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_9_layer_call_and_return_conditional_losses_227841

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
^
B__inference_PoolC3_layer_call_and_return_conditional_losses_230339

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_6_layer_call_and_return_conditional_losses_227772

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????"2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????"*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????"2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????"2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????"2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????"2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????":W S
/
_output_shapes
:?????????"
 
_user_specified_nameinputs
??
?A
__inference__traced_save_231728
file_prefix*
&savev2_cona_kernel_read_readvariableop(
$savev2_cona_bias_read_readvariableop*
&savev2_conc_kernel_read_readvariableop(
$savev2_conc_bias_read_readvariableop*
&savev2_cone_kernel_read_readvariableop(
$savev2_cone_bias_read_readvariableop+
'savev2_cona2_kernel_read_readvariableop)
%savev2_cona2_bias_read_readvariableop+
'savev2_conc2_kernel_read_readvariableop)
%savev2_conc2_bias_read_readvariableop+
'savev2_cone2_kernel_read_readvariableop)
%savev2_cone2_bias_read_readvariableop+
'savev2_cona3_kernel_read_readvariableop)
%savev2_cona3_bias_read_readvariableop+
'savev2_conc3_kernel_read_readvariableop)
%savev2_conc3_bias_read_readvariableop+
'savev2_cone3_kernel_read_readvariableop)
%savev2_cone3_bias_read_readvariableop+
'savev2_cona4_kernel_read_readvariableop)
%savev2_cona4_bias_read_readvariableop+
'savev2_conc4_kernel_read_readvariableop)
%savev2_conc4_bias_read_readvariableop+
'savev2_cone4_kernel_read_readvariableop)
%savev2_cone4_bias_read_readvariableop*
&savev2_hida_kernel_read_readvariableop(
$savev2_hida_bias_read_readvariableop+
'savev2_hida2_kernel_read_readvariableop)
%savev2_hida2_bias_read_readvariableop+
'savev2_hida3_kernel_read_readvariableop)
%savev2_hida3_bias_read_readvariableop+
'savev2_hida4_kernel_read_readvariableop)
%savev2_hida4_bias_read_readvariableop*
&savev2_hidc_kernel_read_readvariableop(
$savev2_hidc_bias_read_readvariableop+
'savev2_hidc2_kernel_read_readvariableop)
%savev2_hidc2_bias_read_readvariableop+
'savev2_hidc3_kernel_read_readvariableop)
%savev2_hidc3_bias_read_readvariableop+
'savev2_hidc4_kernel_read_readvariableop)
%savev2_hidc4_bias_read_readvariableop*
&savev2_hide_kernel_read_readvariableop(
$savev2_hide_bias_read_readvariableop+
'savev2_hide2_kernel_read_readvariableop)
%savev2_hide2_bias_read_readvariableop+
'savev2_hide3_kernel_read_readvariableop)
%savev2_hide3_bias_read_readvariableop+
'savev2_hide4_kernel_read_readvariableop)
%savev2_hide4_bias_read_readvariableop*
&savev2_hid1_kernel_read_readvariableop(
$savev2_hid1_bias_read_readvariableop*
&savev2_hid2_kernel_read_readvariableop(
$savev2_hid2_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop1
-savev2_adam_cona_kernel_m_read_readvariableop/
+savev2_adam_cona_bias_m_read_readvariableop1
-savev2_adam_conc_kernel_m_read_readvariableop/
+savev2_adam_conc_bias_m_read_readvariableop1
-savev2_adam_cone_kernel_m_read_readvariableop/
+savev2_adam_cone_bias_m_read_readvariableop2
.savev2_adam_cona2_kernel_m_read_readvariableop0
,savev2_adam_cona2_bias_m_read_readvariableop2
.savev2_adam_conc2_kernel_m_read_readvariableop0
,savev2_adam_conc2_bias_m_read_readvariableop2
.savev2_adam_cone2_kernel_m_read_readvariableop0
,savev2_adam_cone2_bias_m_read_readvariableop2
.savev2_adam_cona3_kernel_m_read_readvariableop0
,savev2_adam_cona3_bias_m_read_readvariableop2
.savev2_adam_conc3_kernel_m_read_readvariableop0
,savev2_adam_conc3_bias_m_read_readvariableop2
.savev2_adam_cone3_kernel_m_read_readvariableop0
,savev2_adam_cone3_bias_m_read_readvariableop2
.savev2_adam_cona4_kernel_m_read_readvariableop0
,savev2_adam_cona4_bias_m_read_readvariableop2
.savev2_adam_conc4_kernel_m_read_readvariableop0
,savev2_adam_conc4_bias_m_read_readvariableop2
.savev2_adam_cone4_kernel_m_read_readvariableop0
,savev2_adam_cone4_bias_m_read_readvariableop1
-savev2_adam_hida_kernel_m_read_readvariableop/
+savev2_adam_hida_bias_m_read_readvariableop2
.savev2_adam_hida2_kernel_m_read_readvariableop0
,savev2_adam_hida2_bias_m_read_readvariableop2
.savev2_adam_hida3_kernel_m_read_readvariableop0
,savev2_adam_hida3_bias_m_read_readvariableop2
.savev2_adam_hida4_kernel_m_read_readvariableop0
,savev2_adam_hida4_bias_m_read_readvariableop1
-savev2_adam_hidc_kernel_m_read_readvariableop/
+savev2_adam_hidc_bias_m_read_readvariableop2
.savev2_adam_hidc2_kernel_m_read_readvariableop0
,savev2_adam_hidc2_bias_m_read_readvariableop2
.savev2_adam_hidc3_kernel_m_read_readvariableop0
,savev2_adam_hidc3_bias_m_read_readvariableop2
.savev2_adam_hidc4_kernel_m_read_readvariableop0
,savev2_adam_hidc4_bias_m_read_readvariableop1
-savev2_adam_hide_kernel_m_read_readvariableop/
+savev2_adam_hide_bias_m_read_readvariableop2
.savev2_adam_hide2_kernel_m_read_readvariableop0
,savev2_adam_hide2_bias_m_read_readvariableop2
.savev2_adam_hide3_kernel_m_read_readvariableop0
,savev2_adam_hide3_bias_m_read_readvariableop2
.savev2_adam_hide4_kernel_m_read_readvariableop0
,savev2_adam_hide4_bias_m_read_readvariableop1
-savev2_adam_hid1_kernel_m_read_readvariableop/
+savev2_adam_hid1_bias_m_read_readvariableop1
-savev2_adam_hid2_kernel_m_read_readvariableop/
+savev2_adam_hid2_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop1
-savev2_adam_cona_kernel_v_read_readvariableop/
+savev2_adam_cona_bias_v_read_readvariableop1
-savev2_adam_conc_kernel_v_read_readvariableop/
+savev2_adam_conc_bias_v_read_readvariableop1
-savev2_adam_cone_kernel_v_read_readvariableop/
+savev2_adam_cone_bias_v_read_readvariableop2
.savev2_adam_cona2_kernel_v_read_readvariableop0
,savev2_adam_cona2_bias_v_read_readvariableop2
.savev2_adam_conc2_kernel_v_read_readvariableop0
,savev2_adam_conc2_bias_v_read_readvariableop2
.savev2_adam_cone2_kernel_v_read_readvariableop0
,savev2_adam_cone2_bias_v_read_readvariableop2
.savev2_adam_cona3_kernel_v_read_readvariableop0
,savev2_adam_cona3_bias_v_read_readvariableop2
.savev2_adam_conc3_kernel_v_read_readvariableop0
,savev2_adam_conc3_bias_v_read_readvariableop2
.savev2_adam_cone3_kernel_v_read_readvariableop0
,savev2_adam_cone3_bias_v_read_readvariableop2
.savev2_adam_cona4_kernel_v_read_readvariableop0
,savev2_adam_cona4_bias_v_read_readvariableop2
.savev2_adam_conc4_kernel_v_read_readvariableop0
,savev2_adam_conc4_bias_v_read_readvariableop2
.savev2_adam_cone4_kernel_v_read_readvariableop0
,savev2_adam_cone4_bias_v_read_readvariableop1
-savev2_adam_hida_kernel_v_read_readvariableop/
+savev2_adam_hida_bias_v_read_readvariableop2
.savev2_adam_hida2_kernel_v_read_readvariableop0
,savev2_adam_hida2_bias_v_read_readvariableop2
.savev2_adam_hida3_kernel_v_read_readvariableop0
,savev2_adam_hida3_bias_v_read_readvariableop2
.savev2_adam_hida4_kernel_v_read_readvariableop0
,savev2_adam_hida4_bias_v_read_readvariableop1
-savev2_adam_hidc_kernel_v_read_readvariableop/
+savev2_adam_hidc_bias_v_read_readvariableop2
.savev2_adam_hidc2_kernel_v_read_readvariableop0
,savev2_adam_hidc2_bias_v_read_readvariableop2
.savev2_adam_hidc3_kernel_v_read_readvariableop0
,savev2_adam_hidc3_bias_v_read_readvariableop2
.savev2_adam_hidc4_kernel_v_read_readvariableop0
,savev2_adam_hidc4_bias_v_read_readvariableop1
-savev2_adam_hide_kernel_v_read_readvariableop/
+savev2_adam_hide_bias_v_read_readvariableop2
.savev2_adam_hide2_kernel_v_read_readvariableop0
,savev2_adam_hide2_bias_v_read_readvariableop2
.savev2_adam_hide3_kernel_v_read_readvariableop0
,savev2_adam_hide3_bias_v_read_readvariableop2
.savev2_adam_hide4_kernel_v_read_readvariableop0
,savev2_adam_hide4_bias_v_read_readvariableop1
-savev2_adam_hid1_kernel_v_read_readvariableop/
+savev2_adam_hid1_bias_v_read_readvariableop1
-savev2_adam_hid2_kernel_v_read_readvariableop/
+savev2_adam_hid2_bias_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop
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
ShardedFilename?b
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?a
value?aB?a?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-26/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-26/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-26/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-26/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?=
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_cona_kernel_read_readvariableop$savev2_cona_bias_read_readvariableop&savev2_conc_kernel_read_readvariableop$savev2_conc_bias_read_readvariableop&savev2_cone_kernel_read_readvariableop$savev2_cone_bias_read_readvariableop'savev2_cona2_kernel_read_readvariableop%savev2_cona2_bias_read_readvariableop'savev2_conc2_kernel_read_readvariableop%savev2_conc2_bias_read_readvariableop'savev2_cone2_kernel_read_readvariableop%savev2_cone2_bias_read_readvariableop'savev2_cona3_kernel_read_readvariableop%savev2_cona3_bias_read_readvariableop'savev2_conc3_kernel_read_readvariableop%savev2_conc3_bias_read_readvariableop'savev2_cone3_kernel_read_readvariableop%savev2_cone3_bias_read_readvariableop'savev2_cona4_kernel_read_readvariableop%savev2_cona4_bias_read_readvariableop'savev2_conc4_kernel_read_readvariableop%savev2_conc4_bias_read_readvariableop'savev2_cone4_kernel_read_readvariableop%savev2_cone4_bias_read_readvariableop&savev2_hida_kernel_read_readvariableop$savev2_hida_bias_read_readvariableop'savev2_hida2_kernel_read_readvariableop%savev2_hida2_bias_read_readvariableop'savev2_hida3_kernel_read_readvariableop%savev2_hida3_bias_read_readvariableop'savev2_hida4_kernel_read_readvariableop%savev2_hida4_bias_read_readvariableop&savev2_hidc_kernel_read_readvariableop$savev2_hidc_bias_read_readvariableop'savev2_hidc2_kernel_read_readvariableop%savev2_hidc2_bias_read_readvariableop'savev2_hidc3_kernel_read_readvariableop%savev2_hidc3_bias_read_readvariableop'savev2_hidc4_kernel_read_readvariableop%savev2_hidc4_bias_read_readvariableop&savev2_hide_kernel_read_readvariableop$savev2_hide_bias_read_readvariableop'savev2_hide2_kernel_read_readvariableop%savev2_hide2_bias_read_readvariableop'savev2_hide3_kernel_read_readvariableop%savev2_hide3_bias_read_readvariableop'savev2_hide4_kernel_read_readvariableop%savev2_hide4_bias_read_readvariableop&savev2_hid1_kernel_read_readvariableop$savev2_hid1_bias_read_readvariableop&savev2_hid2_kernel_read_readvariableop$savev2_hid2_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop-savev2_adam_cona_kernel_m_read_readvariableop+savev2_adam_cona_bias_m_read_readvariableop-savev2_adam_conc_kernel_m_read_readvariableop+savev2_adam_conc_bias_m_read_readvariableop-savev2_adam_cone_kernel_m_read_readvariableop+savev2_adam_cone_bias_m_read_readvariableop.savev2_adam_cona2_kernel_m_read_readvariableop,savev2_adam_cona2_bias_m_read_readvariableop.savev2_adam_conc2_kernel_m_read_readvariableop,savev2_adam_conc2_bias_m_read_readvariableop.savev2_adam_cone2_kernel_m_read_readvariableop,savev2_adam_cone2_bias_m_read_readvariableop.savev2_adam_cona3_kernel_m_read_readvariableop,savev2_adam_cona3_bias_m_read_readvariableop.savev2_adam_conc3_kernel_m_read_readvariableop,savev2_adam_conc3_bias_m_read_readvariableop.savev2_adam_cone3_kernel_m_read_readvariableop,savev2_adam_cone3_bias_m_read_readvariableop.savev2_adam_cona4_kernel_m_read_readvariableop,savev2_adam_cona4_bias_m_read_readvariableop.savev2_adam_conc4_kernel_m_read_readvariableop,savev2_adam_conc4_bias_m_read_readvariableop.savev2_adam_cone4_kernel_m_read_readvariableop,savev2_adam_cone4_bias_m_read_readvariableop-savev2_adam_hida_kernel_m_read_readvariableop+savev2_adam_hida_bias_m_read_readvariableop.savev2_adam_hida2_kernel_m_read_readvariableop,savev2_adam_hida2_bias_m_read_readvariableop.savev2_adam_hida3_kernel_m_read_readvariableop,savev2_adam_hida3_bias_m_read_readvariableop.savev2_adam_hida4_kernel_m_read_readvariableop,savev2_adam_hida4_bias_m_read_readvariableop-savev2_adam_hidc_kernel_m_read_readvariableop+savev2_adam_hidc_bias_m_read_readvariableop.savev2_adam_hidc2_kernel_m_read_readvariableop,savev2_adam_hidc2_bias_m_read_readvariableop.savev2_adam_hidc3_kernel_m_read_readvariableop,savev2_adam_hidc3_bias_m_read_readvariableop.savev2_adam_hidc4_kernel_m_read_readvariableop,savev2_adam_hidc4_bias_m_read_readvariableop-savev2_adam_hide_kernel_m_read_readvariableop+savev2_adam_hide_bias_m_read_readvariableop.savev2_adam_hide2_kernel_m_read_readvariableop,savev2_adam_hide2_bias_m_read_readvariableop.savev2_adam_hide3_kernel_m_read_readvariableop,savev2_adam_hide3_bias_m_read_readvariableop.savev2_adam_hide4_kernel_m_read_readvariableop,savev2_adam_hide4_bias_m_read_readvariableop-savev2_adam_hid1_kernel_m_read_readvariableop+savev2_adam_hid1_bias_m_read_readvariableop-savev2_adam_hid2_kernel_m_read_readvariableop+savev2_adam_hid2_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop-savev2_adam_cona_kernel_v_read_readvariableop+savev2_adam_cona_bias_v_read_readvariableop-savev2_adam_conc_kernel_v_read_readvariableop+savev2_adam_conc_bias_v_read_readvariableop-savev2_adam_cone_kernel_v_read_readvariableop+savev2_adam_cone_bias_v_read_readvariableop.savev2_adam_cona2_kernel_v_read_readvariableop,savev2_adam_cona2_bias_v_read_readvariableop.savev2_adam_conc2_kernel_v_read_readvariableop,savev2_adam_conc2_bias_v_read_readvariableop.savev2_adam_cone2_kernel_v_read_readvariableop,savev2_adam_cone2_bias_v_read_readvariableop.savev2_adam_cona3_kernel_v_read_readvariableop,savev2_adam_cona3_bias_v_read_readvariableop.savev2_adam_conc3_kernel_v_read_readvariableop,savev2_adam_conc3_bias_v_read_readvariableop.savev2_adam_cone3_kernel_v_read_readvariableop,savev2_adam_cone3_bias_v_read_readvariableop.savev2_adam_cona4_kernel_v_read_readvariableop,savev2_adam_cona4_bias_v_read_readvariableop.savev2_adam_conc4_kernel_v_read_readvariableop,savev2_adam_conc4_bias_v_read_readvariableop.savev2_adam_cone4_kernel_v_read_readvariableop,savev2_adam_cone4_bias_v_read_readvariableop-savev2_adam_hida_kernel_v_read_readvariableop+savev2_adam_hida_bias_v_read_readvariableop.savev2_adam_hida2_kernel_v_read_readvariableop,savev2_adam_hida2_bias_v_read_readvariableop.savev2_adam_hida3_kernel_v_read_readvariableop,savev2_adam_hida3_bias_v_read_readvariableop.savev2_adam_hida4_kernel_v_read_readvariableop,savev2_adam_hida4_bias_v_read_readvariableop-savev2_adam_hidc_kernel_v_read_readvariableop+savev2_adam_hidc_bias_v_read_readvariableop.savev2_adam_hidc2_kernel_v_read_readvariableop,savev2_adam_hidc2_bias_v_read_readvariableop.savev2_adam_hidc3_kernel_v_read_readvariableop,savev2_adam_hidc3_bias_v_read_readvariableop.savev2_adam_hidc4_kernel_v_read_readvariableop,savev2_adam_hidc4_bias_v_read_readvariableop-savev2_adam_hide_kernel_v_read_readvariableop+savev2_adam_hide_bias_v_read_readvariableop.savev2_adam_hide2_kernel_v_read_readvariableop,savev2_adam_hide2_bias_v_read_readvariableop.savev2_adam_hide3_kernel_v_read_readvariableop,savev2_adam_hide3_bias_v_read_readvariableop.savev2_adam_hide4_kernel_v_read_readvariableop,savev2_adam_hide4_bias_v_read_readvariableop-savev2_adam_hid1_kernel_v_read_readvariableop+savev2_adam_hid1_bias_v_read_readvariableop-savev2_adam_hid2_kernel_v_read_readvariableop+savev2_adam_hid2_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes?
?2?	2
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

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :::	$:$:0:0:::$::0::::::::::::::???:?:???:?:
?#?:?:
??:?:??	?:?:???:?:
? ?:?:
??:?:???:?:???:?:
??:?:
??:?:
??:?:
??:?:	?(:(: : : : : : : : : :::	$:$:0:0:::$::0::::::::::::::???:?:???:?:
?#?:?:
??:?:??	?:?:???:?:
? ?:?:
??:?:???:?:???:?:
??:?:
??:?:
??:?:
??:?:	?(:(:::	$:$:0:0:::$::0::::::::::::::???:?:???:?:
?#?:?:
??:?:??	?:?:???:?:
? ?:?:
??:?:???:?:???:?:
??:?:
??:?:
??:?:
??:?:	?(:(: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:	$: 

_output_shapes
:$:,(
&
_output_shapes
:0: 

_output_shapes
:0:,(
&
_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
:$: 


_output_shapes
::,(
&
_output_shapes
:0: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::'#
!
_output_shapes
:???:!

_output_shapes	
:?:'#
!
_output_shapes
:???:!

_output_shapes	
:?:&"
 
_output_shapes
:
?#?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:! 

_output_shapes	
:?:'!#
!
_output_shapes
:??	?:!"

_output_shapes	
:?:'##
!
_output_shapes
:???:!$

_output_shapes	
:?:&%"
 
_output_shapes
:
? ?:!&

_output_shapes	
:?:&'"
 
_output_shapes
:
??:!(

_output_shapes	
:?:')#
!
_output_shapes
:???:!*

_output_shapes	
:?:'+#
!
_output_shapes
:???:!,

_output_shapes	
:?:&-"
 
_output_shapes
:
??:!.

_output_shapes	
:?:&/"
 
_output_shapes
:
??:!0

_output_shapes	
:?:&1"
 
_output_shapes
:
??:!2

_output_shapes	
:?:&3"
 
_output_shapes
:
??:!4

_output_shapes	
:?:%5!

_output_shapes
:	?(: 6

_output_shapes
:(:7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :>

_output_shapes
: :?

_output_shapes
: :,@(
&
_output_shapes
:: A

_output_shapes
::,B(
&
_output_shapes
:	$: C

_output_shapes
:$:,D(
&
_output_shapes
:0: E

_output_shapes
:0:,F(
&
_output_shapes
:: G

_output_shapes
::,H(
&
_output_shapes
:$: I

_output_shapes
::,J(
&
_output_shapes
:0: K

_output_shapes
::,L(
&
_output_shapes
:: M

_output_shapes
::,N(
&
_output_shapes
:: O

_output_shapes
::,P(
&
_output_shapes
:: Q

_output_shapes
::,R(
&
_output_shapes
:: S

_output_shapes
::,T(
&
_output_shapes
:: U

_output_shapes
::,V(
&
_output_shapes
:: W

_output_shapes
::'X#
!
_output_shapes
:???:!Y

_output_shapes	
:?:'Z#
!
_output_shapes
:???:![

_output_shapes	
:?:&\"
 
_output_shapes
:
?#?:!]

_output_shapes	
:?:&^"
 
_output_shapes
:
??:!_

_output_shapes	
:?:'`#
!
_output_shapes
:??	?:!a

_output_shapes	
:?:'b#
!
_output_shapes
:???:!c

_output_shapes	
:?:&d"
 
_output_shapes
:
? ?:!e

_output_shapes	
:?:&f"
 
_output_shapes
:
??:!g

_output_shapes	
:?:'h#
!
_output_shapes
:???:!i

_output_shapes	
:?:'j#
!
_output_shapes
:???:!k

_output_shapes	
:?:&l"
 
_output_shapes
:
??:!m

_output_shapes	
:?:&n"
 
_output_shapes
:
??:!o

_output_shapes	
:?:&p"
 
_output_shapes
:
??:!q

_output_shapes	
:?:&r"
 
_output_shapes
:
??:!s

_output_shapes	
:?:%t!

_output_shapes
:	?(: u

_output_shapes
:(:,v(
&
_output_shapes
:: w

_output_shapes
::,x(
&
_output_shapes
:	$: y

_output_shapes
:$:,z(
&
_output_shapes
:0: {

_output_shapes
:0:,|(
&
_output_shapes
:: }

_output_shapes
::,~(
&
_output_shapes
:$: 

_output_shapes
::-?(
&
_output_shapes
:0:!?

_output_shapes
::-?(
&
_output_shapes
::!?

_output_shapes
::-?(
&
_output_shapes
::!?

_output_shapes
::-?(
&
_output_shapes
::!?

_output_shapes
::-?(
&
_output_shapes
::!?

_output_shapes
::-?(
&
_output_shapes
::!?

_output_shapes
::-?(
&
_output_shapes
::!?

_output_shapes
::(?#
!
_output_shapes
:???:"?

_output_shapes	
:?:(?#
!
_output_shapes
:???:"?

_output_shapes	
:?:'?"
 
_output_shapes
:
?#?:"?

_output_shapes	
:?:'?"
 
_output_shapes
:
??:"?

_output_shapes	
:?:(?#
!
_output_shapes
:??	?:"?

_output_shapes	
:?:(?#
!
_output_shapes
:???:"?

_output_shapes	
:?:'?"
 
_output_shapes
:
? ?:"?

_output_shapes	
:?:'?"
 
_output_shapes
:
??:"?

_output_shapes	
:?:(?#
!
_output_shapes
:???:"?

_output_shapes	
:?:(?#
!
_output_shapes
:???:"?

_output_shapes	
:?:'?"
 
_output_shapes
:
??:"?

_output_shapes	
:?:'?"
 
_output_shapes
:
??:"?

_output_shapes	
:?:'?"
 
_output_shapes
:
??:"?

_output_shapes	
:?:'?"
 
_output_shapes
:
??:"?

_output_shapes	
:?:&?!

_output_shapes
:	?(:!?

_output_shapes
:(:?

_output_shapes
: 
?
F
*__inference_flatten_4_layer_call_fn_230776

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_2270242
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????F=$:W S
/
_output_shapes
:?????????F=$
 
_user_specified_nameinputs
?
^
B__inference_PoolA2_layer_call_and_return_conditional_losses_230199

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
B
&__inference_PoolC_layer_call_fn_230013

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
:?????????F=$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolC_layer_call_and_return_conditional_losses_2266632
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????F=$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????$:Y U
1
_output_shapes
:???????????$
 
_user_specified_nameinputs
?
C
'__inference_PoolA3_layer_call_fn_230314

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA3_layer_call_and_return_conditional_losses_2268282
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????!:W S
/
_output_shapes
:?????????!
 
_user_specified_nameinputs
?
?
B__inference_output_layer_call_and_return_conditional_losses_231192

inputs1
matmul_readvariableop_resource:	?(-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????(2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????(2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
A__inference_HidC4_layer_call_and_return_conditional_losses_231019

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
]
A__inference_PoolA_layer_call_and_return_conditional_losses_226669

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????I?*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????I?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_6_layer_call_fn_230575

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_2277722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????"2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????"22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????"
 
_user_specified_nameinputs
?
]
A__inference_PoolE_layer_call_and_return_conditional_losses_230038

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_8_layer_call_fn_230629

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_2278182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
C
'__inference_PoolE2_layer_call_fn_230234

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
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolE2_layer_call_and_return_conditional_losses_2267472
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@8:W S
/
_output_shapes
:?????????@8
 
_user_specified_nameinputs
?
^
B__inference_PoolE2_layer_call_and_return_conditional_losses_226747

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@8:W S
/
_output_shapes
:?????????@8
 
_user_specified_nameinputs
?
G
+__inference_flatten_11_layer_call_fn_230853

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
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_11_layer_call_and_return_conditional_losses_2269682
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_HidA2_layer_call_fn_230888

inputs
unknown:???
	unknown_0:	?
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
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidA2_layer_call_and_return_conditional_losses_2270862
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
e
F__inference_dropout_11_layer_call_and_return_conditional_losses_230727

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
C
'__inference_PoolE3_layer_call_fn_230349

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
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolE3_layer_call_and_return_conditional_losses_2265162
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_6_layer_call_and_return_conditional_losses_226939

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????"2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????"2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????":W S
/
_output_shapes
:?????????"
 
_user_specified_nameinputs
?
^
B__inference_PoolC4_layer_call_and_return_conditional_losses_226560

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
%__inference_Hid2_layer_call_fn_231161

inputs
unknown:
??
	unknown_0:	?
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
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_Hid2_layer_call_and_return_conditional_losses_2273092
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_9_layer_call_and_return_conditional_losses_230837

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? F  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
]
A__inference_PoolE_layer_call_and_return_conditional_losses_226384

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_226925

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
A__inference_ConA3_layer_call_and_return_conditional_losses_226806

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????!2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????!2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????#
 
_user_specified_nameinputs
?
c
E__inference_dropout_9_layer_call_and_return_conditional_losses_230661

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_230097

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????F=$2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????F=$*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????F=$2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????F=$2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????F=$2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????F=$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????F=$:W S
/
_output_shapes
:?????????F=$
 
_user_specified_nameinputs
?
^
B__inference_PoolA4_layer_call_and_return_conditional_losses_230439

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_5_layer_call_and_return_conditional_losses_227749

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
A__inference_ConE4_layer_call_and_return_conditional_losses_226841

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_5_layer_call_fn_230543

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_2269462
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
A__inference_HidC3_layer_call_and_return_conditional_losses_230999

inputs2
matmul_readvariableop_resource:
? ?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
? ?*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
^
B__inference_PoolE2_layer_call_and_return_conditional_losses_226450

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
+__inference_dropout_10_layer_call_fn_230683

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_2278642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_230760

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????#2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
@__inference_Hid1_layer_call_and_return_conditional_losses_231152

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_230749

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????R  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????#:W S
/
_output_shapes
:?????????#
 
_user_specified_nameinputs
?
?
&__inference_HidE4_layer_call_fn_231088

inputs
unknown:
??
	unknown_0:	?
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
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidE4_layer_call_and_return_conditional_losses_2272562
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_226953

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?*
!__inference__wrapped_model_226331	
inputC
)model_cone_conv2d_readvariableop_resource:08
*model_cone_biasadd_readvariableop_resource:0C
)model_conc_conv2d_readvariableop_resource:	$8
*model_conc_biasadd_readvariableop_resource:$C
)model_cona_conv2d_readvariableop_resource:8
*model_cona_biasadd_readvariableop_resource:D
*model_cone2_conv2d_readvariableop_resource:09
+model_cone2_biasadd_readvariableop_resource:D
*model_conc2_conv2d_readvariableop_resource:$9
+model_conc2_biasadd_readvariableop_resource:D
*model_cona2_conv2d_readvariableop_resource:9
+model_cona2_biasadd_readvariableop_resource:D
*model_cone3_conv2d_readvariableop_resource:9
+model_cone3_biasadd_readvariableop_resource:D
*model_conc3_conv2d_readvariableop_resource:9
+model_conc3_biasadd_readvariableop_resource:D
*model_cona3_conv2d_readvariableop_resource:9
+model_cona3_biasadd_readvariableop_resource:D
*model_cone4_conv2d_readvariableop_resource:9
+model_cone4_biasadd_readvariableop_resource:D
*model_conc4_conv2d_readvariableop_resource:9
+model_conc4_biasadd_readvariableop_resource:D
*model_cona4_conv2d_readvariableop_resource:9
+model_cona4_biasadd_readvariableop_resource:>
)model_hida_matmul_readvariableop_resource:???9
*model_hida_biasadd_readvariableop_resource:	??
*model_hida2_matmul_readvariableop_resource:???:
+model_hida2_biasadd_readvariableop_resource:	?>
*model_hida3_matmul_readvariableop_resource:
?#?:
+model_hida3_biasadd_readvariableop_resource:	?>
*model_hida4_matmul_readvariableop_resource:
??:
+model_hida4_biasadd_readvariableop_resource:	?>
)model_hidc_matmul_readvariableop_resource:??	?9
*model_hidc_biasadd_readvariableop_resource:	??
*model_hidc2_matmul_readvariableop_resource:???:
+model_hidc2_biasadd_readvariableop_resource:	?>
*model_hidc3_matmul_readvariableop_resource:
? ?:
+model_hidc3_biasadd_readvariableop_resource:	?>
*model_hidc4_matmul_readvariableop_resource:
??:
+model_hidc4_biasadd_readvariableop_resource:	?>
)model_hide_matmul_readvariableop_resource:???9
*model_hide_biasadd_readvariableop_resource:	??
*model_hide2_matmul_readvariableop_resource:???:
+model_hide2_biasadd_readvariableop_resource:	?>
*model_hide3_matmul_readvariableop_resource:
??:
+model_hide3_biasadd_readvariableop_resource:	?>
*model_hide4_matmul_readvariableop_resource:
??:
+model_hide4_biasadd_readvariableop_resource:	?=
)model_hid1_matmul_readvariableop_resource:
??9
*model_hid1_biasadd_readvariableop_resource:	?=
)model_hid2_matmul_readvariableop_resource:
??9
*model_hid2_biasadd_readvariableop_resource:	?>
+model_output_matmul_readvariableop_resource:	?(:
,model_output_biasadd_readvariableop_resource:(
identity??!model/ConA/BiasAdd/ReadVariableOp? model/ConA/Conv2D/ReadVariableOp?"model/ConA2/BiasAdd/ReadVariableOp?!model/ConA2/Conv2D/ReadVariableOp?"model/ConA3/BiasAdd/ReadVariableOp?!model/ConA3/Conv2D/ReadVariableOp?"model/ConA4/BiasAdd/ReadVariableOp?!model/ConA4/Conv2D/ReadVariableOp?!model/ConC/BiasAdd/ReadVariableOp? model/ConC/Conv2D/ReadVariableOp?"model/ConC2/BiasAdd/ReadVariableOp?!model/ConC2/Conv2D/ReadVariableOp?"model/ConC3/BiasAdd/ReadVariableOp?!model/ConC3/Conv2D/ReadVariableOp?"model/ConC4/BiasAdd/ReadVariableOp?!model/ConC4/Conv2D/ReadVariableOp?!model/ConE/BiasAdd/ReadVariableOp? model/ConE/Conv2D/ReadVariableOp?"model/ConE2/BiasAdd/ReadVariableOp?!model/ConE2/Conv2D/ReadVariableOp?"model/ConE3/BiasAdd/ReadVariableOp?!model/ConE3/Conv2D/ReadVariableOp?"model/ConE4/BiasAdd/ReadVariableOp?!model/ConE4/Conv2D/ReadVariableOp?!model/Hid1/BiasAdd/ReadVariableOp? model/Hid1/MatMul/ReadVariableOp?!model/Hid2/BiasAdd/ReadVariableOp? model/Hid2/MatMul/ReadVariableOp?!model/HidA/BiasAdd/ReadVariableOp? model/HidA/MatMul/ReadVariableOp?"model/HidA2/BiasAdd/ReadVariableOp?!model/HidA2/MatMul/ReadVariableOp?"model/HidA3/BiasAdd/ReadVariableOp?!model/HidA3/MatMul/ReadVariableOp?"model/HidA4/BiasAdd/ReadVariableOp?!model/HidA4/MatMul/ReadVariableOp?!model/HidC/BiasAdd/ReadVariableOp? model/HidC/MatMul/ReadVariableOp?"model/HidC2/BiasAdd/ReadVariableOp?!model/HidC2/MatMul/ReadVariableOp?"model/HidC3/BiasAdd/ReadVariableOp?!model/HidC3/MatMul/ReadVariableOp?"model/HidC4/BiasAdd/ReadVariableOp?!model/HidC4/MatMul/ReadVariableOp?!model/HidE/BiasAdd/ReadVariableOp? model/HidE/MatMul/ReadVariableOp?"model/HidE2/BiasAdd/ReadVariableOp?!model/HidE2/MatMul/ReadVariableOp?"model/HidE3/BiasAdd/ReadVariableOp?!model/HidE3/MatMul/ReadVariableOp?"model/HidE4/BiasAdd/ReadVariableOp?!model/HidE4/MatMul/ReadVariableOp?#model/output/BiasAdd/ReadVariableOp?"model/output/MatMul/ReadVariableOp?
 model/ConE/Conv2D/ReadVariableOpReadVariableOp)model_cone_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02"
 model/ConE/Conv2D/ReadVariableOp?
model/ConE/Conv2DConv2Dinput(model/ConE/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingVALID*
strides
2
model/ConE/Conv2D?
!model/ConE/BiasAdd/ReadVariableOpReadVariableOp*model_cone_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02#
!model/ConE/BiasAdd/ReadVariableOp?
model/ConE/BiasAddBiasAddmodel/ConE/Conv2D:output:0)model/ConE/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02
model/ConE/BiasAdd?
model/ConE/ReluRelumodel/ConE/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02
model/ConE/Relu?
 model/ConC/Conv2D/ReadVariableOpReadVariableOp)model_conc_conv2d_readvariableop_resource*&
_output_shapes
:	$*
dtype02"
 model/ConC/Conv2D/ReadVariableOp?
model/ConC/Conv2DConv2Dinput(model/ConC/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$*
paddingVALID*
strides
2
model/ConC/Conv2D?
!model/ConC/BiasAdd/ReadVariableOpReadVariableOp*model_conc_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02#
!model/ConC/BiasAdd/ReadVariableOp?
model/ConC/BiasAddBiasAddmodel/ConC/Conv2D:output:0)model/ConC/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$2
model/ConC/BiasAdd?
model/ConC/ReluRelumodel/ConC/BiasAdd:output:0*
T0*1
_output_shapes
:???????????$2
model/ConC/Relu?
 model/ConA/Conv2D/ReadVariableOpReadVariableOp)model_cona_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 model/ConA/Conv2D/ReadVariableOp?
model/ConA/Conv2DConv2Dinput(model/ConA/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
model/ConA/Conv2D?
!model/ConA/BiasAdd/ReadVariableOpReadVariableOp*model_cona_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!model/ConA/BiasAdd/ReadVariableOp?
model/ConA/BiasAddBiasAddmodel/ConA/Conv2D:output:0)model/ConA/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
model/ConA/BiasAdd?
model/ConA/ReluRelumodel/ConA/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
model/ConA/Relu?
model/PoolE/MaxPoolMaxPoolmodel/ConE/Relu:activations:0*/
_output_shapes
:?????????B:0*
ksize
*
paddingVALID*
strides
2
model/PoolE/MaxPool?
model/PoolC/MaxPoolMaxPoolmodel/ConC/Relu:activations:0*/
_output_shapes
:?????????F=$*
ksize
*
paddingVALID*
strides
2
model/PoolC/MaxPool?
model/PoolA/MaxPoolMaxPoolmodel/ConA/Relu:activations:0*/
_output_shapes
:?????????I?*
ksize
*
paddingVALID*
strides
2
model/PoolA/MaxPool?
model/dropout_2/IdentityIdentitymodel/PoolE/MaxPool:output:0*
T0*/
_output_shapes
:?????????B:02
model/dropout_2/Identity?
model/dropout_1/IdentityIdentitymodel/PoolC/MaxPool:output:0*
T0*/
_output_shapes
:?????????F=$2
model/dropout_1/Identity?
model/dropout/IdentityIdentitymodel/PoolA/MaxPool:output:0*
T0*/
_output_shapes
:?????????I?2
model/dropout/Identity?
!model/ConE2/Conv2D/ReadVariableOpReadVariableOp*model_cone2_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02#
!model/ConE2/Conv2D/ReadVariableOp?
model/ConE2/Conv2DConv2D!model/dropout_2/Identity:output:0)model/ConE2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@8*
paddingVALID*
strides
2
model/ConE2/Conv2D?
"model/ConE2/BiasAdd/ReadVariableOpReadVariableOp+model_cone2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/ConE2/BiasAdd/ReadVariableOp?
model/ConE2/BiasAddBiasAddmodel/ConE2/Conv2D:output:0*model/ConE2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@82
model/ConE2/BiasAdd?
model/ConE2/ReluRelumodel/ConE2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@82
model/ConE2/Relu?
!model/ConC2/Conv2D/ReadVariableOpReadVariableOp*model_conc2_conv2d_readvariableop_resource*&
_output_shapes
:$*
dtype02#
!model/ConC2/Conv2D/ReadVariableOp?
model/ConC2/Conv2DConv2D!model/dropout_1/Identity:output:0)model/ConC2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????D;*
paddingVALID*
strides
2
model/ConC2/Conv2D?
"model/ConC2/BiasAdd/ReadVariableOpReadVariableOp+model_conc2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/ConC2/BiasAdd/ReadVariableOp?
model/ConC2/BiasAddBiasAddmodel/ConC2/Conv2D:output:0*model/ConC2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????D;2
model/ConC2/BiasAdd?
model/ConC2/ReluRelumodel/ConC2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????D;2
model/ConC2/Relu?
!model/ConA2/Conv2D/ReadVariableOpReadVariableOp*model_cona2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02#
!model/ConA2/Conv2D/ReadVariableOp?
model/ConA2/Conv2DConv2Dmodel/dropout/Identity:output:0)model/ConA2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????G=*
paddingVALID*
strides
2
model/ConA2/Conv2D?
"model/ConA2/BiasAdd/ReadVariableOpReadVariableOp+model_cona2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/ConA2/BiasAdd/ReadVariableOp?
model/ConA2/BiasAddBiasAddmodel/ConA2/Conv2D:output:0*model/ConA2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????G=2
model/ConA2/BiasAdd?
model/ConA2/ReluRelumodel/ConA2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????G=2
model/ConA2/Relu?
model/PoolE2/MaxPoolMaxPoolmodel/ConE2/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
model/PoolE2/MaxPool?
model/PoolC2/MaxPoolMaxPoolmodel/ConC2/Relu:activations:0*/
_output_shapes
:?????????"*
ksize
*
paddingVALID*
strides
2
model/PoolC2/MaxPool?
model/PoolA2/MaxPoolMaxPoolmodel/ConA2/Relu:activations:0*/
_output_shapes
:?????????#*
ksize
*
paddingVALID*
strides
2
model/PoolA2/MaxPool?
!model/ConE3/Conv2D/ReadVariableOpReadVariableOp*model_cone3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02#
!model/ConE3/Conv2D/ReadVariableOp?
model/ConE3/Conv2DConv2Dmodel/PoolE2/MaxPool:output:0)model/ConE3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model/ConE3/Conv2D?
"model/ConE3/BiasAdd/ReadVariableOpReadVariableOp+model_cone3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/ConE3/BiasAdd/ReadVariableOp?
model/ConE3/BiasAddBiasAddmodel/ConE3/Conv2D:output:0*model/ConE3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model/ConE3/BiasAdd?
model/ConE3/ReluRelumodel/ConE3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model/ConE3/Relu?
!model/ConC3/Conv2D/ReadVariableOpReadVariableOp*model_conc3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02#
!model/ConC3/Conv2D/ReadVariableOp?
model/ConC3/Conv2DConv2Dmodel/PoolC2/MaxPool:output:0)model/ConC3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
model/ConC3/Conv2D?
"model/ConC3/BiasAdd/ReadVariableOpReadVariableOp+model_conc3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/ConC3/BiasAdd/ReadVariableOp?
model/ConC3/BiasAddBiasAddmodel/ConC3/Conv2D:output:0*model/ConC3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
model/ConC3/BiasAdd?
model/ConC3/ReluRelumodel/ConC3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
model/ConC3/Relu?
!model/ConA3/Conv2D/ReadVariableOpReadVariableOp*model_cona3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02#
!model/ConA3/Conv2D/ReadVariableOp?
model/ConA3/Conv2DConv2Dmodel/PoolA2/MaxPool:output:0)model/ConA3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!*
paddingVALID*
strides
2
model/ConA3/Conv2D?
"model/ConA3/BiasAdd/ReadVariableOpReadVariableOp+model_cona3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/ConA3/BiasAdd/ReadVariableOp?
model/ConA3/BiasAddBiasAddmodel/ConA3/Conv2D:output:0*model/ConA3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!2
model/ConA3/BiasAdd?
model/ConA3/ReluRelumodel/ConA3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????!2
model/ConA3/Relu?
model/PoolE3/MaxPoolMaxPoolmodel/ConE3/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
model/PoolE3/MaxPool?
model/PoolC3/MaxPoolMaxPoolmodel/ConC3/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
model/PoolC3/MaxPool?
model/PoolA3/MaxPoolMaxPoolmodel/ConA3/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
model/PoolA3/MaxPool?
!model/ConE4/Conv2D/ReadVariableOpReadVariableOp*model_cone4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02#
!model/ConE4/Conv2D/ReadVariableOp?
model/ConE4/Conv2DConv2Dmodel/PoolE3/MaxPool:output:0)model/ConE4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model/ConE4/Conv2D?
"model/ConE4/BiasAdd/ReadVariableOpReadVariableOp+model_cone4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/ConE4/BiasAdd/ReadVariableOp?
model/ConE4/BiasAddBiasAddmodel/ConE4/Conv2D:output:0*model/ConE4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model/ConE4/BiasAdd?
model/ConE4/ReluRelumodel/ConE4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model/ConE4/Relu?
!model/ConC4/Conv2D/ReadVariableOpReadVariableOp*model_conc4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02#
!model/ConC4/Conv2D/ReadVariableOp?
model/ConC4/Conv2DConv2Dmodel/PoolC3/MaxPool:output:0)model/ConC4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model/ConC4/Conv2D?
"model/ConC4/BiasAdd/ReadVariableOpReadVariableOp+model_conc4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/ConC4/BiasAdd/ReadVariableOp?
model/ConC4/BiasAddBiasAddmodel/ConC4/Conv2D:output:0*model/ConC4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model/ConC4/BiasAdd?
model/ConC4/ReluRelumodel/ConC4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model/ConC4/Relu?
!model/ConA4/Conv2D/ReadVariableOpReadVariableOp*model_cona4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02#
!model/ConA4/Conv2D/ReadVariableOp?
model/ConA4/Conv2DConv2Dmodel/PoolA3/MaxPool:output:0)model/ConA4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model/ConA4/Conv2D?
"model/ConA4/BiasAdd/ReadVariableOpReadVariableOp+model_cona4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/ConA4/BiasAdd/ReadVariableOp?
model/ConA4/BiasAddBiasAddmodel/ConA4/Conv2D:output:0*model/ConA4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model/ConA4/BiasAdd?
model/ConA4/ReluRelumodel/ConA4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model/ConA4/Relu?
model/PoolE4/MaxPoolMaxPoolmodel/ConE4/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
model/PoolE4/MaxPool?
model/PoolC4/MaxPoolMaxPoolmodel/ConC4/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
model/PoolC4/MaxPool?
model/PoolA4/MaxPoolMaxPoolmodel/ConA4/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
model/PoolA4/MaxPool?
model/dropout_11/IdentityIdentitymodel/PoolE4/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
model/dropout_11/Identity?
model/dropout_10/IdentityIdentitymodel/PoolE3/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
model/dropout_10/Identity?
model/dropout_9/IdentityIdentitymodel/PoolE2/MaxPool:output:0*
T0*/
_output_shapes
:????????? 2
model/dropout_9/Identity?
model/dropout_8/IdentityIdentitymodel/PoolC4/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
model/dropout_8/Identity?
model/dropout_7/IdentityIdentitymodel/PoolC3/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
model/dropout_7/Identity?
model/dropout_6/IdentityIdentitymodel/PoolC2/MaxPool:output:0*
T0*/
_output_shapes
:?????????"2
model/dropout_6/Identity?
model/dropout_5/IdentityIdentitymodel/PoolA4/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
model/dropout_5/Identity?
model/dropout_4/IdentityIdentitymodel/PoolA3/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
model/dropout_4/Identity?
model/dropout_3/IdentityIdentitymodel/PoolA2/MaxPool:output:0*
T0*/
_output_shapes
:?????????#2
model/dropout_3/Identity?
model/flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"????X  2
model/flatten_11/Const?
model/flatten_11/ReshapeReshape"model/dropout_11/Identity:output:0model/flatten_11/Const:output:0*
T0*(
_output_shapes
:??????????2
model/flatten_11/Reshape?
model/flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"????<  2
model/flatten_10/Const?
model/flatten_10/ReshapeReshape"model/dropout_10/Identity:output:0model/flatten_10/Const:output:0*
T0*(
_output_shapes
:??????????2
model/flatten_10/Reshape
model/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? F  2
model/flatten_9/Const?
model/flatten_9/ReshapeReshape!model/dropout_9/Identity:output:0model/flatten_9/Const:output:0*
T0*)
_output_shapes
:???????????2
model/flatten_9/Reshape
model/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????? 2
model/flatten_8/Const?
model/flatten_8/ReshapeReshapemodel/PoolE/MaxPool:output:0model/flatten_8/Const:output:0*
T0*)
_output_shapes
:???????????2
model/flatten_8/Reshape
model/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
model/flatten_7/Const?
model/flatten_7/ReshapeReshape!model/dropout_8/Identity:output:0model/flatten_7/Const:output:0*
T0*(
_output_shapes
:??????????2
model/flatten_7/Reshape
model/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
model/flatten_6/Const?
model/flatten_6/ReshapeReshape!model/dropout_7/Identity:output:0model/flatten_6/Const:output:0*
T0*(
_output_shapes
:?????????? 2
model/flatten_6/Reshape
model/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????M  2
model/flatten_5/Const?
model/flatten_5/ReshapeReshape!model/dropout_6/Identity:output:0model/flatten_5/Const:output:0*
T0*)
_output_shapes
:???????????2
model/flatten_5/Reshape
model/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????xX 2
model/flatten_4/Const?
model/flatten_4/ReshapeReshapemodel/PoolC/MaxPool:output:0model/flatten_4/Const:output:0*
T0*)
_output_shapes
:???????????	2
model/flatten_4/Reshape
model/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????H  2
model/flatten_3/Const?
model/flatten_3/ReshapeReshape!model/dropout_5/Identity:output:0model/flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????2
model/flatten_3/Reshape
model/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
model/flatten_2/Const?
model/flatten_2/ReshapeReshape!model/dropout_4/Identity:output:0model/flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????#2
model/flatten_2/Reshape
model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????R  2
model/flatten_1/Const?
model/flatten_1/ReshapeReshape!model/dropout_3/Identity:output:0model/flatten_1/Const:output:0*
T0*)
_output_shapes
:???????????2
model/flatten_1/Reshape{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????(? 2
model/flatten/Const?
model/flatten/ReshapeReshapemodel/PoolA/MaxPool:output:0model/flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
model/flatten/Reshape?
 model/HidA/MatMul/ReadVariableOpReadVariableOp)model_hida_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02"
 model/HidA/MatMul/ReadVariableOp?
model/HidA/MatMulMatMulmodel/flatten/Reshape:output:0(model/HidA/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidA/MatMul?
!model/HidA/BiasAdd/ReadVariableOpReadVariableOp*model_hida_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!model/HidA/BiasAdd/ReadVariableOp?
model/HidA/BiasAddBiasAddmodel/HidA/MatMul:product:0)model/HidA/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidA/BiasAddz
model/HidA/ReluRelumodel/HidA/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidA/Relu?
!model/HidA2/MatMul/ReadVariableOpReadVariableOp*model_hida2_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02#
!model/HidA2/MatMul/ReadVariableOp?
model/HidA2/MatMulMatMul model/flatten_1/Reshape:output:0)model/HidA2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidA2/MatMul?
"model/HidA2/BiasAdd/ReadVariableOpReadVariableOp+model_hida2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"model/HidA2/BiasAdd/ReadVariableOp?
model/HidA2/BiasAddBiasAddmodel/HidA2/MatMul:product:0*model/HidA2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidA2/BiasAdd}
model/HidA2/ReluRelumodel/HidA2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidA2/Relu?
!model/HidA3/MatMul/ReadVariableOpReadVariableOp*model_hida3_matmul_readvariableop_resource* 
_output_shapes
:
?#?*
dtype02#
!model/HidA3/MatMul/ReadVariableOp?
model/HidA3/MatMulMatMul model/flatten_2/Reshape:output:0)model/HidA3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidA3/MatMul?
"model/HidA3/BiasAdd/ReadVariableOpReadVariableOp+model_hida3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"model/HidA3/BiasAdd/ReadVariableOp?
model/HidA3/BiasAddBiasAddmodel/HidA3/MatMul:product:0*model/HidA3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidA3/BiasAdd}
model/HidA3/ReluRelumodel/HidA3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidA3/Relu?
!model/HidA4/MatMul/ReadVariableOpReadVariableOp*model_hida4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!model/HidA4/MatMul/ReadVariableOp?
model/HidA4/MatMulMatMul model/flatten_3/Reshape:output:0)model/HidA4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidA4/MatMul?
"model/HidA4/BiasAdd/ReadVariableOpReadVariableOp+model_hida4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"model/HidA4/BiasAdd/ReadVariableOp?
model/HidA4/BiasAddBiasAddmodel/HidA4/MatMul:product:0*model/HidA4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidA4/BiasAdd}
model/HidA4/ReluRelumodel/HidA4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidA4/Relu?
 model/HidC/MatMul/ReadVariableOpReadVariableOp)model_hidc_matmul_readvariableop_resource*!
_output_shapes
:??	?*
dtype02"
 model/HidC/MatMul/ReadVariableOp?
model/HidC/MatMulMatMul model/flatten_4/Reshape:output:0(model/HidC/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidC/MatMul?
!model/HidC/BiasAdd/ReadVariableOpReadVariableOp*model_hidc_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!model/HidC/BiasAdd/ReadVariableOp?
model/HidC/BiasAddBiasAddmodel/HidC/MatMul:product:0)model/HidC/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidC/BiasAddz
model/HidC/ReluRelumodel/HidC/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidC/Relu?
!model/HidC2/MatMul/ReadVariableOpReadVariableOp*model_hidc2_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02#
!model/HidC2/MatMul/ReadVariableOp?
model/HidC2/MatMulMatMul model/flatten_5/Reshape:output:0)model/HidC2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidC2/MatMul?
"model/HidC2/BiasAdd/ReadVariableOpReadVariableOp+model_hidc2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"model/HidC2/BiasAdd/ReadVariableOp?
model/HidC2/BiasAddBiasAddmodel/HidC2/MatMul:product:0*model/HidC2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidC2/BiasAdd}
model/HidC2/ReluRelumodel/HidC2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidC2/Relu?
!model/HidC3/MatMul/ReadVariableOpReadVariableOp*model_hidc3_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02#
!model/HidC3/MatMul/ReadVariableOp?
model/HidC3/MatMulMatMul model/flatten_6/Reshape:output:0)model/HidC3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidC3/MatMul?
"model/HidC3/BiasAdd/ReadVariableOpReadVariableOp+model_hidc3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"model/HidC3/BiasAdd/ReadVariableOp?
model/HidC3/BiasAddBiasAddmodel/HidC3/MatMul:product:0*model/HidC3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidC3/BiasAdd}
model/HidC3/ReluRelumodel/HidC3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidC3/Relu?
!model/HidC4/MatMul/ReadVariableOpReadVariableOp*model_hidc4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!model/HidC4/MatMul/ReadVariableOp?
model/HidC4/MatMulMatMul model/flatten_7/Reshape:output:0)model/HidC4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidC4/MatMul?
"model/HidC4/BiasAdd/ReadVariableOpReadVariableOp+model_hidc4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"model/HidC4/BiasAdd/ReadVariableOp?
model/HidC4/BiasAddBiasAddmodel/HidC4/MatMul:product:0*model/HidC4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidC4/BiasAdd}
model/HidC4/ReluRelumodel/HidC4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidC4/Relu?
 model/HidE/MatMul/ReadVariableOpReadVariableOp)model_hide_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02"
 model/HidE/MatMul/ReadVariableOp?
model/HidE/MatMulMatMul model/flatten_8/Reshape:output:0(model/HidE/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidE/MatMul?
!model/HidE/BiasAdd/ReadVariableOpReadVariableOp*model_hide_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!model/HidE/BiasAdd/ReadVariableOp?
model/HidE/BiasAddBiasAddmodel/HidE/MatMul:product:0)model/HidE/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidE/BiasAddz
model/HidE/ReluRelumodel/HidE/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidE/Relu?
!model/HidE2/MatMul/ReadVariableOpReadVariableOp*model_hide2_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02#
!model/HidE2/MatMul/ReadVariableOp?
model/HidE2/MatMulMatMul model/flatten_9/Reshape:output:0)model/HidE2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidE2/MatMul?
"model/HidE2/BiasAdd/ReadVariableOpReadVariableOp+model_hide2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"model/HidE2/BiasAdd/ReadVariableOp?
model/HidE2/BiasAddBiasAddmodel/HidE2/MatMul:product:0*model/HidE2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidE2/BiasAdd}
model/HidE2/ReluRelumodel/HidE2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidE2/Relu?
!model/HidE3/MatMul/ReadVariableOpReadVariableOp*model_hide3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!model/HidE3/MatMul/ReadVariableOp?
model/HidE3/MatMulMatMul!model/flatten_10/Reshape:output:0)model/HidE3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidE3/MatMul?
"model/HidE3/BiasAdd/ReadVariableOpReadVariableOp+model_hide3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"model/HidE3/BiasAdd/ReadVariableOp?
model/HidE3/BiasAddBiasAddmodel/HidE3/MatMul:product:0*model/HidE3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidE3/BiasAdd}
model/HidE3/ReluRelumodel/HidE3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidE3/Relu?
!model/HidE4/MatMul/ReadVariableOpReadVariableOp*model_hide4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!model/HidE4/MatMul/ReadVariableOp?
model/HidE4/MatMulMatMul!model/flatten_11/Reshape:output:0)model/HidE4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidE4/MatMul?
"model/HidE4/BiasAdd/ReadVariableOpReadVariableOp+model_hide4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"model/HidE4/BiasAdd/ReadVariableOp?
model/HidE4/BiasAddBiasAddmodel/HidE4/MatMul:product:0*model/HidE4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidE4/BiasAdd}
model/HidE4/ReluRelumodel/HidE4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidE4/Relu?
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axis?
model/concatenate/concatConcatV2model/HidA/Relu:activations:0model/HidA2/Relu:activations:0model/HidA3/Relu:activations:0model/HidA4/Relu:activations:0model/HidC/Relu:activations:0model/HidC2/Relu:activations:0model/HidC3/Relu:activations:0model/HidC4/Relu:activations:0model/HidE/Relu:activations:0model/HidE2/Relu:activations:0model/HidE3/Relu:activations:0model/HidE4/Relu:activations:0&model/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
model/concatenate/concat?
 model/Hid1/MatMul/ReadVariableOpReadVariableOp)model_hid1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 model/Hid1/MatMul/ReadVariableOp?
model/Hid1/MatMulMatMul!model/concatenate/concat:output:0(model/Hid1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/Hid1/MatMul?
!model/Hid1/BiasAdd/ReadVariableOpReadVariableOp*model_hid1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!model/Hid1/BiasAdd/ReadVariableOp?
model/Hid1/BiasAddBiasAddmodel/Hid1/MatMul:product:0)model/Hid1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/Hid1/BiasAddz
model/Hid1/ReluRelumodel/Hid1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/Hid1/Relu?
 model/Hid2/MatMul/ReadVariableOpReadVariableOp)model_hid2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 model/Hid2/MatMul/ReadVariableOp?
model/Hid2/MatMulMatMulmodel/Hid1/Relu:activations:0(model/Hid2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/Hid2/MatMul?
!model/Hid2/BiasAdd/ReadVariableOpReadVariableOp*model_hid2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!model/Hid2/BiasAdd/ReadVariableOp?
model/Hid2/BiasAddBiasAddmodel/Hid2/MatMul:product:0)model/Hid2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/Hid2/BiasAddz
model/Hid2/TanhTanhmodel/Hid2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/Hid2/Tanh?
"model/output/MatMul/ReadVariableOpReadVariableOp+model_output_matmul_readvariableop_resource*
_output_shapes
:	?(*
dtype02$
"model/output/MatMul/ReadVariableOp?
model/output/MatMulMatMulmodel/Hid2/Tanh:y:0*model/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model/output/MatMul?
#model/output/BiasAdd/ReadVariableOpReadVariableOp,model_output_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02%
#model/output/BiasAdd/ReadVariableOp?
model/output/BiasAddBiasAddmodel/output/MatMul:product:0+model/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model/output/BiasAdd?
model/output/SoftmaxSoftmaxmodel/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model/output/Softmaxy
IdentityIdentitymodel/output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????(2

Identity?
NoOpNoOp"^model/ConA/BiasAdd/ReadVariableOp!^model/ConA/Conv2D/ReadVariableOp#^model/ConA2/BiasAdd/ReadVariableOp"^model/ConA2/Conv2D/ReadVariableOp#^model/ConA3/BiasAdd/ReadVariableOp"^model/ConA3/Conv2D/ReadVariableOp#^model/ConA4/BiasAdd/ReadVariableOp"^model/ConA4/Conv2D/ReadVariableOp"^model/ConC/BiasAdd/ReadVariableOp!^model/ConC/Conv2D/ReadVariableOp#^model/ConC2/BiasAdd/ReadVariableOp"^model/ConC2/Conv2D/ReadVariableOp#^model/ConC3/BiasAdd/ReadVariableOp"^model/ConC3/Conv2D/ReadVariableOp#^model/ConC4/BiasAdd/ReadVariableOp"^model/ConC4/Conv2D/ReadVariableOp"^model/ConE/BiasAdd/ReadVariableOp!^model/ConE/Conv2D/ReadVariableOp#^model/ConE2/BiasAdd/ReadVariableOp"^model/ConE2/Conv2D/ReadVariableOp#^model/ConE3/BiasAdd/ReadVariableOp"^model/ConE3/Conv2D/ReadVariableOp#^model/ConE4/BiasAdd/ReadVariableOp"^model/ConE4/Conv2D/ReadVariableOp"^model/Hid1/BiasAdd/ReadVariableOp!^model/Hid1/MatMul/ReadVariableOp"^model/Hid2/BiasAdd/ReadVariableOp!^model/Hid2/MatMul/ReadVariableOp"^model/HidA/BiasAdd/ReadVariableOp!^model/HidA/MatMul/ReadVariableOp#^model/HidA2/BiasAdd/ReadVariableOp"^model/HidA2/MatMul/ReadVariableOp#^model/HidA3/BiasAdd/ReadVariableOp"^model/HidA3/MatMul/ReadVariableOp#^model/HidA4/BiasAdd/ReadVariableOp"^model/HidA4/MatMul/ReadVariableOp"^model/HidC/BiasAdd/ReadVariableOp!^model/HidC/MatMul/ReadVariableOp#^model/HidC2/BiasAdd/ReadVariableOp"^model/HidC2/MatMul/ReadVariableOp#^model/HidC3/BiasAdd/ReadVariableOp"^model/HidC3/MatMul/ReadVariableOp#^model/HidC4/BiasAdd/ReadVariableOp"^model/HidC4/MatMul/ReadVariableOp"^model/HidE/BiasAdd/ReadVariableOp!^model/HidE/MatMul/ReadVariableOp#^model/HidE2/BiasAdd/ReadVariableOp"^model/HidE2/MatMul/ReadVariableOp#^model/HidE3/BiasAdd/ReadVariableOp"^model/HidE3/MatMul/ReadVariableOp#^model/HidE4/BiasAdd/ReadVariableOp"^model/HidE4/MatMul/ReadVariableOp$^model/output/BiasAdd/ReadVariableOp#^model/output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!model/ConA/BiasAdd/ReadVariableOp!model/ConA/BiasAdd/ReadVariableOp2D
 model/ConA/Conv2D/ReadVariableOp model/ConA/Conv2D/ReadVariableOp2H
"model/ConA2/BiasAdd/ReadVariableOp"model/ConA2/BiasAdd/ReadVariableOp2F
!model/ConA2/Conv2D/ReadVariableOp!model/ConA2/Conv2D/ReadVariableOp2H
"model/ConA3/BiasAdd/ReadVariableOp"model/ConA3/BiasAdd/ReadVariableOp2F
!model/ConA3/Conv2D/ReadVariableOp!model/ConA3/Conv2D/ReadVariableOp2H
"model/ConA4/BiasAdd/ReadVariableOp"model/ConA4/BiasAdd/ReadVariableOp2F
!model/ConA4/Conv2D/ReadVariableOp!model/ConA4/Conv2D/ReadVariableOp2F
!model/ConC/BiasAdd/ReadVariableOp!model/ConC/BiasAdd/ReadVariableOp2D
 model/ConC/Conv2D/ReadVariableOp model/ConC/Conv2D/ReadVariableOp2H
"model/ConC2/BiasAdd/ReadVariableOp"model/ConC2/BiasAdd/ReadVariableOp2F
!model/ConC2/Conv2D/ReadVariableOp!model/ConC2/Conv2D/ReadVariableOp2H
"model/ConC3/BiasAdd/ReadVariableOp"model/ConC3/BiasAdd/ReadVariableOp2F
!model/ConC3/Conv2D/ReadVariableOp!model/ConC3/Conv2D/ReadVariableOp2H
"model/ConC4/BiasAdd/ReadVariableOp"model/ConC4/BiasAdd/ReadVariableOp2F
!model/ConC4/Conv2D/ReadVariableOp!model/ConC4/Conv2D/ReadVariableOp2F
!model/ConE/BiasAdd/ReadVariableOp!model/ConE/BiasAdd/ReadVariableOp2D
 model/ConE/Conv2D/ReadVariableOp model/ConE/Conv2D/ReadVariableOp2H
"model/ConE2/BiasAdd/ReadVariableOp"model/ConE2/BiasAdd/ReadVariableOp2F
!model/ConE2/Conv2D/ReadVariableOp!model/ConE2/Conv2D/ReadVariableOp2H
"model/ConE3/BiasAdd/ReadVariableOp"model/ConE3/BiasAdd/ReadVariableOp2F
!model/ConE3/Conv2D/ReadVariableOp!model/ConE3/Conv2D/ReadVariableOp2H
"model/ConE4/BiasAdd/ReadVariableOp"model/ConE4/BiasAdd/ReadVariableOp2F
!model/ConE4/Conv2D/ReadVariableOp!model/ConE4/Conv2D/ReadVariableOp2F
!model/Hid1/BiasAdd/ReadVariableOp!model/Hid1/BiasAdd/ReadVariableOp2D
 model/Hid1/MatMul/ReadVariableOp model/Hid1/MatMul/ReadVariableOp2F
!model/Hid2/BiasAdd/ReadVariableOp!model/Hid2/BiasAdd/ReadVariableOp2D
 model/Hid2/MatMul/ReadVariableOp model/Hid2/MatMul/ReadVariableOp2F
!model/HidA/BiasAdd/ReadVariableOp!model/HidA/BiasAdd/ReadVariableOp2D
 model/HidA/MatMul/ReadVariableOp model/HidA/MatMul/ReadVariableOp2H
"model/HidA2/BiasAdd/ReadVariableOp"model/HidA2/BiasAdd/ReadVariableOp2F
!model/HidA2/MatMul/ReadVariableOp!model/HidA2/MatMul/ReadVariableOp2H
"model/HidA3/BiasAdd/ReadVariableOp"model/HidA3/BiasAdd/ReadVariableOp2F
!model/HidA3/MatMul/ReadVariableOp!model/HidA3/MatMul/ReadVariableOp2H
"model/HidA4/BiasAdd/ReadVariableOp"model/HidA4/BiasAdd/ReadVariableOp2F
!model/HidA4/MatMul/ReadVariableOp!model/HidA4/MatMul/ReadVariableOp2F
!model/HidC/BiasAdd/ReadVariableOp!model/HidC/BiasAdd/ReadVariableOp2D
 model/HidC/MatMul/ReadVariableOp model/HidC/MatMul/ReadVariableOp2H
"model/HidC2/BiasAdd/ReadVariableOp"model/HidC2/BiasAdd/ReadVariableOp2F
!model/HidC2/MatMul/ReadVariableOp!model/HidC2/MatMul/ReadVariableOp2H
"model/HidC3/BiasAdd/ReadVariableOp"model/HidC3/BiasAdd/ReadVariableOp2F
!model/HidC3/MatMul/ReadVariableOp!model/HidC3/MatMul/ReadVariableOp2H
"model/HidC4/BiasAdd/ReadVariableOp"model/HidC4/BiasAdd/ReadVariableOp2F
!model/HidC4/MatMul/ReadVariableOp!model/HidC4/MatMul/ReadVariableOp2F
!model/HidE/BiasAdd/ReadVariableOp!model/HidE/BiasAdd/ReadVariableOp2D
 model/HidE/MatMul/ReadVariableOp model/HidE/MatMul/ReadVariableOp2H
"model/HidE2/BiasAdd/ReadVariableOp"model/HidE2/BiasAdd/ReadVariableOp2F
!model/HidE2/MatMul/ReadVariableOp!model/HidE2/MatMul/ReadVariableOp2H
"model/HidE3/BiasAdd/ReadVariableOp"model/HidE3/BiasAdd/ReadVariableOp2F
!model/HidE3/MatMul/ReadVariableOp!model/HidE3/MatMul/ReadVariableOp2H
"model/HidE4/BiasAdd/ReadVariableOp"model/HidE4/BiasAdd/ReadVariableOp2F
!model/HidE4/MatMul/ReadVariableOp!model/HidE4/MatMul/ReadVariableOp2J
#model/output/BiasAdd/ReadVariableOp#model/output/BiasAdd/ReadVariableOp2H
"model/output/MatMul/ReadVariableOp"model/output/MatMul/ReadVariableOp:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
d
E__inference_dropout_8_layer_call_and_return_conditional_losses_227818

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_7_layer_call_and_return_conditional_losses_230607

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_6_layer_call_and_return_conditional_losses_227008

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????? 2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_7_layer_call_fn_230809

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
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_2270002
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
@__inference_HidC_layer_call_and_return_conditional_losses_230959

inputs3
matmul_readvariableop_resource:??	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:??	?*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????	
 
_user_specified_nameinputs
?
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_230634

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_HidC4_layer_call_fn_231008

inputs
unknown:
??
	unknown_0:	?
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
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidC4_layer_call_and_return_conditional_losses_2271882
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_flatten_10_layer_call_fn_230842

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
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_10_layer_call_and_return_conditional_losses_2269762
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
@__inference_ConA_layer_call_and_return_conditional_losses_226647

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_230526

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_227444	
input!
unknown:0
	unknown_0:0#
	unknown_1:	$
	unknown_2:$#
	unknown_3:
	unknown_4:#
	unknown_5:0
	unknown_6:#
	unknown_7:$
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:$

unknown_15:

unknown_16:$

unknown_17:

unknown_18:$

unknown_19:

unknown_20:$

unknown_21:

unknown_22:

unknown_23:???

unknown_24:	?

unknown_25:???

unknown_26:	?

unknown_27:
?#?

unknown_28:	?

unknown_29:
??

unknown_30:	?

unknown_31:??	?

unknown_32:	?

unknown_33:???

unknown_34:	?

unknown_35:
? ?

unknown_36:	?

unknown_37:
??

unknown_38:	?

unknown_39:???

unknown_40:	?

unknown_41:???

unknown_42:	?

unknown_43:
??

unknown_44:	?

unknown_45:
??

unknown_46:	?

unknown_47:
??

unknown_48:	?

unknown_49:
??

unknown_50:	?

unknown_51:	?(

unknown_52:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2273332
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_227032

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????H  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
]
A__inference_PoolA_layer_call_and_return_conditional_losses_226340

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_7_layer_call_and_return_conditional_losses_230815

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
A__inference_HidE4_layer_call_and_return_conditional_losses_227256

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
(__inference_dropout_layer_call_fn_230053

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_2280452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????I?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????I?22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????I?
 
_user_specified_nameinputs
?
G
+__inference_dropout_10_layer_call_fn_230678

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_2269112
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_PoolE3_layer_call_and_return_conditional_losses_230364

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
F__inference_dropout_11_layer_call_and_return_conditional_losses_227887

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?%
A__inference_model_layer_call_and_return_conditional_losses_229596

inputs=
#cone_conv2d_readvariableop_resource:02
$cone_biasadd_readvariableop_resource:0=
#conc_conv2d_readvariableop_resource:	$2
$conc_biasadd_readvariableop_resource:$=
#cona_conv2d_readvariableop_resource:2
$cona_biasadd_readvariableop_resource:>
$cone2_conv2d_readvariableop_resource:03
%cone2_biasadd_readvariableop_resource:>
$conc2_conv2d_readvariableop_resource:$3
%conc2_biasadd_readvariableop_resource:>
$cona2_conv2d_readvariableop_resource:3
%cona2_biasadd_readvariableop_resource:>
$cone3_conv2d_readvariableop_resource:3
%cone3_biasadd_readvariableop_resource:>
$conc3_conv2d_readvariableop_resource:3
%conc3_biasadd_readvariableop_resource:>
$cona3_conv2d_readvariableop_resource:3
%cona3_biasadd_readvariableop_resource:>
$cone4_conv2d_readvariableop_resource:3
%cone4_biasadd_readvariableop_resource:>
$conc4_conv2d_readvariableop_resource:3
%conc4_biasadd_readvariableop_resource:>
$cona4_conv2d_readvariableop_resource:3
%cona4_biasadd_readvariableop_resource:8
#hida_matmul_readvariableop_resource:???3
$hida_biasadd_readvariableop_resource:	?9
$hida2_matmul_readvariableop_resource:???4
%hida2_biasadd_readvariableop_resource:	?8
$hida3_matmul_readvariableop_resource:
?#?4
%hida3_biasadd_readvariableop_resource:	?8
$hida4_matmul_readvariableop_resource:
??4
%hida4_biasadd_readvariableop_resource:	?8
#hidc_matmul_readvariableop_resource:??	?3
$hidc_biasadd_readvariableop_resource:	?9
$hidc2_matmul_readvariableop_resource:???4
%hidc2_biasadd_readvariableop_resource:	?8
$hidc3_matmul_readvariableop_resource:
? ?4
%hidc3_biasadd_readvariableop_resource:	?8
$hidc4_matmul_readvariableop_resource:
??4
%hidc4_biasadd_readvariableop_resource:	?8
#hide_matmul_readvariableop_resource:???3
$hide_biasadd_readvariableop_resource:	?9
$hide2_matmul_readvariableop_resource:???4
%hide2_biasadd_readvariableop_resource:	?8
$hide3_matmul_readvariableop_resource:
??4
%hide3_biasadd_readvariableop_resource:	?8
$hide4_matmul_readvariableop_resource:
??4
%hide4_biasadd_readvariableop_resource:	?7
#hid1_matmul_readvariableop_resource:
??3
$hid1_biasadd_readvariableop_resource:	?7
#hid2_matmul_readvariableop_resource:
??3
$hid2_biasadd_readvariableop_resource:	?8
%output_matmul_readvariableop_resource:	?(4
&output_biasadd_readvariableop_resource:(
identity??ConA/BiasAdd/ReadVariableOp?ConA/Conv2D/ReadVariableOp?ConA2/BiasAdd/ReadVariableOp?ConA2/Conv2D/ReadVariableOp?ConA3/BiasAdd/ReadVariableOp?ConA3/Conv2D/ReadVariableOp?ConA4/BiasAdd/ReadVariableOp?ConA4/Conv2D/ReadVariableOp?ConC/BiasAdd/ReadVariableOp?ConC/Conv2D/ReadVariableOp?ConC2/BiasAdd/ReadVariableOp?ConC2/Conv2D/ReadVariableOp?ConC3/BiasAdd/ReadVariableOp?ConC3/Conv2D/ReadVariableOp?ConC4/BiasAdd/ReadVariableOp?ConC4/Conv2D/ReadVariableOp?ConE/BiasAdd/ReadVariableOp?ConE/Conv2D/ReadVariableOp?ConE2/BiasAdd/ReadVariableOp?ConE2/Conv2D/ReadVariableOp?ConE3/BiasAdd/ReadVariableOp?ConE3/Conv2D/ReadVariableOp?ConE4/BiasAdd/ReadVariableOp?ConE4/Conv2D/ReadVariableOp?Hid1/BiasAdd/ReadVariableOp?Hid1/MatMul/ReadVariableOp?Hid2/BiasAdd/ReadVariableOp?Hid2/MatMul/ReadVariableOp?HidA/BiasAdd/ReadVariableOp?HidA/MatMul/ReadVariableOp?HidA2/BiasAdd/ReadVariableOp?HidA2/MatMul/ReadVariableOp?HidA3/BiasAdd/ReadVariableOp?HidA3/MatMul/ReadVariableOp?HidA4/BiasAdd/ReadVariableOp?HidA4/MatMul/ReadVariableOp?HidC/BiasAdd/ReadVariableOp?HidC/MatMul/ReadVariableOp?HidC2/BiasAdd/ReadVariableOp?HidC2/MatMul/ReadVariableOp?HidC3/BiasAdd/ReadVariableOp?HidC3/MatMul/ReadVariableOp?HidC4/BiasAdd/ReadVariableOp?HidC4/MatMul/ReadVariableOp?HidE/BiasAdd/ReadVariableOp?HidE/MatMul/ReadVariableOp?HidE2/BiasAdd/ReadVariableOp?HidE2/MatMul/ReadVariableOp?HidE3/BiasAdd/ReadVariableOp?HidE3/MatMul/ReadVariableOp?HidE4/BiasAdd/ReadVariableOp?HidE4/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
ConE/Conv2D/ReadVariableOpReadVariableOp#cone_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02
ConE/Conv2D/ReadVariableOp?
ConE/Conv2DConv2Dinputs"ConE/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingVALID*
strides
2
ConE/Conv2D?
ConE/BiasAdd/ReadVariableOpReadVariableOp$cone_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
ConE/BiasAdd/ReadVariableOp?
ConE/BiasAddBiasAddConE/Conv2D:output:0#ConE/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02
ConE/BiasAddq
	ConE/ReluReluConE/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02
	ConE/Relu?
ConC/Conv2D/ReadVariableOpReadVariableOp#conc_conv2d_readvariableop_resource*&
_output_shapes
:	$*
dtype02
ConC/Conv2D/ReadVariableOp?
ConC/Conv2DConv2Dinputs"ConC/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$*
paddingVALID*
strides
2
ConC/Conv2D?
ConC/BiasAdd/ReadVariableOpReadVariableOp$conc_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
ConC/BiasAdd/ReadVariableOp?
ConC/BiasAddBiasAddConC/Conv2D:output:0#ConC/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$2
ConC/BiasAddq
	ConC/ReluReluConC/BiasAdd:output:0*
T0*1
_output_shapes
:???????????$2
	ConC/Relu?
ConA/Conv2D/ReadVariableOpReadVariableOp#cona_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConA/Conv2D/ReadVariableOp?
ConA/Conv2DConv2Dinputs"ConA/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
ConA/Conv2D?
ConA/BiasAdd/ReadVariableOpReadVariableOp$cona_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConA/BiasAdd/ReadVariableOp?
ConA/BiasAddBiasAddConA/Conv2D:output:0#ConA/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
ConA/BiasAddq
	ConA/ReluReluConA/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
	ConA/Relu?
PoolE/MaxPoolMaxPoolConE/Relu:activations:0*/
_output_shapes
:?????????B:0*
ksize
*
paddingVALID*
strides
2
PoolE/MaxPool?
PoolC/MaxPoolMaxPoolConC/Relu:activations:0*/
_output_shapes
:?????????F=$*
ksize
*
paddingVALID*
strides
2
PoolC/MaxPool?
PoolA/MaxPoolMaxPoolConA/Relu:activations:0*/
_output_shapes
:?????????I?*
ksize
*
paddingVALID*
strides
2
PoolA/MaxPool?
dropout_2/IdentityIdentityPoolE/MaxPool:output:0*
T0*/
_output_shapes
:?????????B:02
dropout_2/Identity?
dropout_1/IdentityIdentityPoolC/MaxPool:output:0*
T0*/
_output_shapes
:?????????F=$2
dropout_1/Identity?
dropout/IdentityIdentityPoolA/MaxPool:output:0*
T0*/
_output_shapes
:?????????I?2
dropout/Identity?
ConE2/Conv2D/ReadVariableOpReadVariableOp$cone2_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02
ConE2/Conv2D/ReadVariableOp?
ConE2/Conv2DConv2Ddropout_2/Identity:output:0#ConE2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@8*
paddingVALID*
strides
2
ConE2/Conv2D?
ConE2/BiasAdd/ReadVariableOpReadVariableOp%cone2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConE2/BiasAdd/ReadVariableOp?
ConE2/BiasAddBiasAddConE2/Conv2D:output:0$ConE2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@82
ConE2/BiasAddr

ConE2/ReluReluConE2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@82

ConE2/Relu?
ConC2/Conv2D/ReadVariableOpReadVariableOp$conc2_conv2d_readvariableop_resource*&
_output_shapes
:$*
dtype02
ConC2/Conv2D/ReadVariableOp?
ConC2/Conv2DConv2Ddropout_1/Identity:output:0#ConC2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????D;*
paddingVALID*
strides
2
ConC2/Conv2D?
ConC2/BiasAdd/ReadVariableOpReadVariableOp%conc2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConC2/BiasAdd/ReadVariableOp?
ConC2/BiasAddBiasAddConC2/Conv2D:output:0$ConC2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????D;2
ConC2/BiasAddr

ConC2/ReluReluConC2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????D;2

ConC2/Relu?
ConA2/Conv2D/ReadVariableOpReadVariableOp$cona2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConA2/Conv2D/ReadVariableOp?
ConA2/Conv2DConv2Ddropout/Identity:output:0#ConA2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????G=*
paddingVALID*
strides
2
ConA2/Conv2D?
ConA2/BiasAdd/ReadVariableOpReadVariableOp%cona2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConA2/BiasAdd/ReadVariableOp?
ConA2/BiasAddBiasAddConA2/Conv2D:output:0$ConA2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????G=2
ConA2/BiasAddr

ConA2/ReluReluConA2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????G=2

ConA2/Relu?
PoolE2/MaxPoolMaxPoolConE2/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
PoolE2/MaxPool?
PoolC2/MaxPoolMaxPoolConC2/Relu:activations:0*/
_output_shapes
:?????????"*
ksize
*
paddingVALID*
strides
2
PoolC2/MaxPool?
PoolA2/MaxPoolMaxPoolConA2/Relu:activations:0*/
_output_shapes
:?????????#*
ksize
*
paddingVALID*
strides
2
PoolA2/MaxPool?
ConE3/Conv2D/ReadVariableOpReadVariableOp$cone3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConE3/Conv2D/ReadVariableOp?
ConE3/Conv2DConv2DPoolE2/MaxPool:output:0#ConE3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
ConE3/Conv2D?
ConE3/BiasAdd/ReadVariableOpReadVariableOp%cone3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConE3/BiasAdd/ReadVariableOp?
ConE3/BiasAddBiasAddConE3/Conv2D:output:0$ConE3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
ConE3/BiasAddr

ConE3/ReluReluConE3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2

ConE3/Relu?
ConC3/Conv2D/ReadVariableOpReadVariableOp$conc3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConC3/Conv2D/ReadVariableOp?
ConC3/Conv2DConv2DPoolC2/MaxPool:output:0#ConC3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
ConC3/Conv2D?
ConC3/BiasAdd/ReadVariableOpReadVariableOp%conc3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConC3/BiasAdd/ReadVariableOp?
ConC3/BiasAddBiasAddConC3/Conv2D:output:0$ConC3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
ConC3/BiasAddr

ConC3/ReluReluConC3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2

ConC3/Relu?
ConA3/Conv2D/ReadVariableOpReadVariableOp$cona3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConA3/Conv2D/ReadVariableOp?
ConA3/Conv2DConv2DPoolA2/MaxPool:output:0#ConA3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!*
paddingVALID*
strides
2
ConA3/Conv2D?
ConA3/BiasAdd/ReadVariableOpReadVariableOp%cona3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConA3/BiasAdd/ReadVariableOp?
ConA3/BiasAddBiasAddConA3/Conv2D:output:0$ConA3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!2
ConA3/BiasAddr

ConA3/ReluReluConA3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????!2

ConA3/Relu?
PoolE3/MaxPoolMaxPoolConE3/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
PoolE3/MaxPool?
PoolC3/MaxPoolMaxPoolConC3/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
PoolC3/MaxPool?
PoolA3/MaxPoolMaxPoolConA3/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
PoolA3/MaxPool?
ConE4/Conv2D/ReadVariableOpReadVariableOp$cone4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConE4/Conv2D/ReadVariableOp?
ConE4/Conv2DConv2DPoolE3/MaxPool:output:0#ConE4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
ConE4/Conv2D?
ConE4/BiasAdd/ReadVariableOpReadVariableOp%cone4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConE4/BiasAdd/ReadVariableOp?
ConE4/BiasAddBiasAddConE4/Conv2D:output:0$ConE4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
ConE4/BiasAddr

ConE4/ReluReluConE4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2

ConE4/Relu?
ConC4/Conv2D/ReadVariableOpReadVariableOp$conc4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConC4/Conv2D/ReadVariableOp?
ConC4/Conv2DConv2DPoolC3/MaxPool:output:0#ConC4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
ConC4/Conv2D?
ConC4/BiasAdd/ReadVariableOpReadVariableOp%conc4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConC4/BiasAdd/ReadVariableOp?
ConC4/BiasAddBiasAddConC4/Conv2D:output:0$ConC4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
ConC4/BiasAddr

ConC4/ReluReluConC4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2

ConC4/Relu?
ConA4/Conv2D/ReadVariableOpReadVariableOp$cona4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConA4/Conv2D/ReadVariableOp?
ConA4/Conv2DConv2DPoolA3/MaxPool:output:0#ConA4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
ConA4/Conv2D?
ConA4/BiasAdd/ReadVariableOpReadVariableOp%cona4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConA4/BiasAdd/ReadVariableOp?
ConA4/BiasAddBiasAddConA4/Conv2D:output:0$ConA4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
ConA4/BiasAddr

ConA4/ReluReluConA4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2

ConA4/Relu?
PoolE4/MaxPoolMaxPoolConE4/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
PoolE4/MaxPool?
PoolC4/MaxPoolMaxPoolConC4/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
PoolC4/MaxPool?
PoolA4/MaxPoolMaxPoolConA4/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
PoolA4/MaxPool?
dropout_11/IdentityIdentityPoolE4/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
dropout_11/Identity?
dropout_10/IdentityIdentityPoolE3/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
dropout_10/Identity?
dropout_9/IdentityIdentityPoolE2/MaxPool:output:0*
T0*/
_output_shapes
:????????? 2
dropout_9/Identity?
dropout_8/IdentityIdentityPoolC4/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
dropout_8/Identity?
dropout_7/IdentityIdentityPoolC3/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
dropout_7/Identity?
dropout_6/IdentityIdentityPoolC2/MaxPool:output:0*
T0*/
_output_shapes
:?????????"2
dropout_6/Identity?
dropout_5/IdentityIdentityPoolA4/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
dropout_5/Identity?
dropout_4/IdentityIdentityPoolA3/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
dropout_4/Identity?
dropout_3/IdentityIdentityPoolA2/MaxPool:output:0*
T0*/
_output_shapes
:?????????#2
dropout_3/Identityu
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"????X  2
flatten_11/Const?
flatten_11/ReshapeReshapedropout_11/Identity:output:0flatten_11/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_11/Reshapeu
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"????<  2
flatten_10/Const?
flatten_10/ReshapeReshapedropout_10/Identity:output:0flatten_10/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_10/Reshapes
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? F  2
flatten_9/Const?
flatten_9/ReshapeReshapedropout_9/Identity:output:0flatten_9/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_9/Reshapes
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????? 2
flatten_8/Const?
flatten_8/ReshapeReshapePoolE/MaxPool:output:0flatten_8/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_8/Reshapes
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_7/Const?
flatten_7/ReshapeReshapedropout_8/Identity:output:0flatten_7/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_7/Reshapes
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
flatten_6/Const?
flatten_6/ReshapeReshapedropout_7/Identity:output:0flatten_6/Const:output:0*
T0*(
_output_shapes
:?????????? 2
flatten_6/Reshapes
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????M  2
flatten_5/Const?
flatten_5/ReshapeReshapedropout_6/Identity:output:0flatten_5/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_5/Reshapes
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????xX 2
flatten_4/Const?
flatten_4/ReshapeReshapePoolC/MaxPool:output:0flatten_4/Const:output:0*
T0*)
_output_shapes
:???????????	2
flatten_4/Reshapes
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????H  2
flatten_3/Const?
flatten_3/ReshapeReshapedropout_5/Identity:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_3/Reshapes
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_2/Const?
flatten_2/ReshapeReshapedropout_4/Identity:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????#2
flatten_2/Reshapes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????R  2
flatten_1/Const?
flatten_1/ReshapeReshapedropout_3/Identity:output:0flatten_1/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_1/Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????(? 2
flatten/Const?
flatten/ReshapeReshapePoolA/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten/Reshape?
HidA/MatMul/ReadVariableOpReadVariableOp#hida_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
HidA/MatMul/ReadVariableOp?
HidA/MatMulMatMulflatten/Reshape:output:0"HidA/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA/MatMul?
HidA/BiasAdd/ReadVariableOpReadVariableOp$hida_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidA/BiasAdd/ReadVariableOp?
HidA/BiasAddBiasAddHidA/MatMul:product:0#HidA/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA/BiasAddh
	HidA/ReluReluHidA/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
	HidA/Relu?
HidA2/MatMul/ReadVariableOpReadVariableOp$hida2_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
HidA2/MatMul/ReadVariableOp?
HidA2/MatMulMatMulflatten_1/Reshape:output:0#HidA2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA2/MatMul?
HidA2/BiasAdd/ReadVariableOpReadVariableOp%hida2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidA2/BiasAdd/ReadVariableOp?
HidA2/BiasAddBiasAddHidA2/MatMul:product:0$HidA2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA2/BiasAddk

HidA2/ReluReluHidA2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidA2/Relu?
HidA3/MatMul/ReadVariableOpReadVariableOp$hida3_matmul_readvariableop_resource* 
_output_shapes
:
?#?*
dtype02
HidA3/MatMul/ReadVariableOp?
HidA3/MatMulMatMulflatten_2/Reshape:output:0#HidA3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA3/MatMul?
HidA3/BiasAdd/ReadVariableOpReadVariableOp%hida3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidA3/BiasAdd/ReadVariableOp?
HidA3/BiasAddBiasAddHidA3/MatMul:product:0$HidA3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA3/BiasAddk

HidA3/ReluReluHidA3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidA3/Relu?
HidA4/MatMul/ReadVariableOpReadVariableOp$hida4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
HidA4/MatMul/ReadVariableOp?
HidA4/MatMulMatMulflatten_3/Reshape:output:0#HidA4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA4/MatMul?
HidA4/BiasAdd/ReadVariableOpReadVariableOp%hida4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidA4/BiasAdd/ReadVariableOp?
HidA4/BiasAddBiasAddHidA4/MatMul:product:0$HidA4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA4/BiasAddk

HidA4/ReluReluHidA4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidA4/Relu?
HidC/MatMul/ReadVariableOpReadVariableOp#hidc_matmul_readvariableop_resource*!
_output_shapes
:??	?*
dtype02
HidC/MatMul/ReadVariableOp?
HidC/MatMulMatMulflatten_4/Reshape:output:0"HidC/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidC/MatMul?
HidC/BiasAdd/ReadVariableOpReadVariableOp$hidc_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidC/BiasAdd/ReadVariableOp?
HidC/BiasAddBiasAddHidC/MatMul:product:0#HidC/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidC/BiasAddh
	HidC/ReluReluHidC/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
	HidC/Relu?
HidC2/MatMul/ReadVariableOpReadVariableOp$hidc2_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
HidC2/MatMul/ReadVariableOp?
HidC2/MatMulMatMulflatten_5/Reshape:output:0#HidC2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidC2/MatMul?
HidC2/BiasAdd/ReadVariableOpReadVariableOp%hidc2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidC2/BiasAdd/ReadVariableOp?
HidC2/BiasAddBiasAddHidC2/MatMul:product:0$HidC2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidC2/BiasAddk

HidC2/ReluReluHidC2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidC2/Relu?
HidC3/MatMul/ReadVariableOpReadVariableOp$hidc3_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02
HidC3/MatMul/ReadVariableOp?
HidC3/MatMulMatMulflatten_6/Reshape:output:0#HidC3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidC3/MatMul?
HidC3/BiasAdd/ReadVariableOpReadVariableOp%hidc3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidC3/BiasAdd/ReadVariableOp?
HidC3/BiasAddBiasAddHidC3/MatMul:product:0$HidC3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidC3/BiasAddk

HidC3/ReluReluHidC3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidC3/Relu?
HidC4/MatMul/ReadVariableOpReadVariableOp$hidc4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
HidC4/MatMul/ReadVariableOp?
HidC4/MatMulMatMulflatten_7/Reshape:output:0#HidC4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidC4/MatMul?
HidC4/BiasAdd/ReadVariableOpReadVariableOp%hidc4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidC4/BiasAdd/ReadVariableOp?
HidC4/BiasAddBiasAddHidC4/MatMul:product:0$HidC4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidC4/BiasAddk

HidC4/ReluReluHidC4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidC4/Relu?
HidE/MatMul/ReadVariableOpReadVariableOp#hide_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
HidE/MatMul/ReadVariableOp?
HidE/MatMulMatMulflatten_8/Reshape:output:0"HidE/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidE/MatMul?
HidE/BiasAdd/ReadVariableOpReadVariableOp$hide_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidE/BiasAdd/ReadVariableOp?
HidE/BiasAddBiasAddHidE/MatMul:product:0#HidE/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidE/BiasAddh
	HidE/ReluReluHidE/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
	HidE/Relu?
HidE2/MatMul/ReadVariableOpReadVariableOp$hide2_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
HidE2/MatMul/ReadVariableOp?
HidE2/MatMulMatMulflatten_9/Reshape:output:0#HidE2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidE2/MatMul?
HidE2/BiasAdd/ReadVariableOpReadVariableOp%hide2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidE2/BiasAdd/ReadVariableOp?
HidE2/BiasAddBiasAddHidE2/MatMul:product:0$HidE2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidE2/BiasAddk

HidE2/ReluReluHidE2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidE2/Relu?
HidE3/MatMul/ReadVariableOpReadVariableOp$hide3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
HidE3/MatMul/ReadVariableOp?
HidE3/MatMulMatMulflatten_10/Reshape:output:0#HidE3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidE3/MatMul?
HidE3/BiasAdd/ReadVariableOpReadVariableOp%hide3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidE3/BiasAdd/ReadVariableOp?
HidE3/BiasAddBiasAddHidE3/MatMul:product:0$HidE3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidE3/BiasAddk

HidE3/ReluReluHidE3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidE3/Relu?
HidE4/MatMul/ReadVariableOpReadVariableOp$hide4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
HidE4/MatMul/ReadVariableOp?
HidE4/MatMulMatMulflatten_11/Reshape:output:0#HidE4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidE4/MatMul?
HidE4/BiasAdd/ReadVariableOpReadVariableOp%hide4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidE4/BiasAdd/ReadVariableOp?
HidE4/BiasAddBiasAddHidE4/MatMul:product:0$HidE4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidE4/BiasAddk

HidE4/ReluReluHidE4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidE4/Relut
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2HidA/Relu:activations:0HidA2/Relu:activations:0HidA3/Relu:activations:0HidA4/Relu:activations:0HidC/Relu:activations:0HidC2/Relu:activations:0HidC3/Relu:activations:0HidC4/Relu:activations:0HidE/Relu:activations:0HidE2/Relu:activations:0HidE3/Relu:activations:0HidE4/Relu:activations:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate/concat?
Hid1/MatMul/ReadVariableOpReadVariableOp#hid1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Hid1/MatMul/ReadVariableOp?
Hid1/MatMulMatMulconcatenate/concat:output:0"Hid1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Hid1/MatMul?
Hid1/BiasAdd/ReadVariableOpReadVariableOp$hid1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
Hid1/BiasAdd/ReadVariableOp?
Hid1/BiasAddBiasAddHid1/MatMul:product:0#Hid1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Hid1/BiasAddh
	Hid1/ReluReluHid1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
	Hid1/Relu?
Hid2/MatMul/ReadVariableOpReadVariableOp#hid2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Hid2/MatMul/ReadVariableOp?
Hid2/MatMulMatMulHid1/Relu:activations:0"Hid2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Hid2/MatMul?
Hid2/BiasAdd/ReadVariableOpReadVariableOp$hid2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
Hid2/BiasAdd/ReadVariableOp?
Hid2/BiasAddBiasAddHid2/MatMul:product:0#Hid2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Hid2/BiasAddh
	Hid2/TanhTanhHid2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
	Hid2/Tanh?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	?(*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMulHid2/Tanh:y:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
output/Softmaxs
IdentityIdentityoutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????(2

Identity?
NoOpNoOp^ConA/BiasAdd/ReadVariableOp^ConA/Conv2D/ReadVariableOp^ConA2/BiasAdd/ReadVariableOp^ConA2/Conv2D/ReadVariableOp^ConA3/BiasAdd/ReadVariableOp^ConA3/Conv2D/ReadVariableOp^ConA4/BiasAdd/ReadVariableOp^ConA4/Conv2D/ReadVariableOp^ConC/BiasAdd/ReadVariableOp^ConC/Conv2D/ReadVariableOp^ConC2/BiasAdd/ReadVariableOp^ConC2/Conv2D/ReadVariableOp^ConC3/BiasAdd/ReadVariableOp^ConC3/Conv2D/ReadVariableOp^ConC4/BiasAdd/ReadVariableOp^ConC4/Conv2D/ReadVariableOp^ConE/BiasAdd/ReadVariableOp^ConE/Conv2D/ReadVariableOp^ConE2/BiasAdd/ReadVariableOp^ConE2/Conv2D/ReadVariableOp^ConE3/BiasAdd/ReadVariableOp^ConE3/Conv2D/ReadVariableOp^ConE4/BiasAdd/ReadVariableOp^ConE4/Conv2D/ReadVariableOp^Hid1/BiasAdd/ReadVariableOp^Hid1/MatMul/ReadVariableOp^Hid2/BiasAdd/ReadVariableOp^Hid2/MatMul/ReadVariableOp^HidA/BiasAdd/ReadVariableOp^HidA/MatMul/ReadVariableOp^HidA2/BiasAdd/ReadVariableOp^HidA2/MatMul/ReadVariableOp^HidA3/BiasAdd/ReadVariableOp^HidA3/MatMul/ReadVariableOp^HidA4/BiasAdd/ReadVariableOp^HidA4/MatMul/ReadVariableOp^HidC/BiasAdd/ReadVariableOp^HidC/MatMul/ReadVariableOp^HidC2/BiasAdd/ReadVariableOp^HidC2/MatMul/ReadVariableOp^HidC3/BiasAdd/ReadVariableOp^HidC3/MatMul/ReadVariableOp^HidC4/BiasAdd/ReadVariableOp^HidC4/MatMul/ReadVariableOp^HidE/BiasAdd/ReadVariableOp^HidE/MatMul/ReadVariableOp^HidE2/BiasAdd/ReadVariableOp^HidE2/MatMul/ReadVariableOp^HidE3/BiasAdd/ReadVariableOp^HidE3/MatMul/ReadVariableOp^HidE4/BiasAdd/ReadVariableOp^HidE4/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
ConA/BiasAdd/ReadVariableOpConA/BiasAdd/ReadVariableOp28
ConA/Conv2D/ReadVariableOpConA/Conv2D/ReadVariableOp2<
ConA2/BiasAdd/ReadVariableOpConA2/BiasAdd/ReadVariableOp2:
ConA2/Conv2D/ReadVariableOpConA2/Conv2D/ReadVariableOp2<
ConA3/BiasAdd/ReadVariableOpConA3/BiasAdd/ReadVariableOp2:
ConA3/Conv2D/ReadVariableOpConA3/Conv2D/ReadVariableOp2<
ConA4/BiasAdd/ReadVariableOpConA4/BiasAdd/ReadVariableOp2:
ConA4/Conv2D/ReadVariableOpConA4/Conv2D/ReadVariableOp2:
ConC/BiasAdd/ReadVariableOpConC/BiasAdd/ReadVariableOp28
ConC/Conv2D/ReadVariableOpConC/Conv2D/ReadVariableOp2<
ConC2/BiasAdd/ReadVariableOpConC2/BiasAdd/ReadVariableOp2:
ConC2/Conv2D/ReadVariableOpConC2/Conv2D/ReadVariableOp2<
ConC3/BiasAdd/ReadVariableOpConC3/BiasAdd/ReadVariableOp2:
ConC3/Conv2D/ReadVariableOpConC3/Conv2D/ReadVariableOp2<
ConC4/BiasAdd/ReadVariableOpConC4/BiasAdd/ReadVariableOp2:
ConC4/Conv2D/ReadVariableOpConC4/Conv2D/ReadVariableOp2:
ConE/BiasAdd/ReadVariableOpConE/BiasAdd/ReadVariableOp28
ConE/Conv2D/ReadVariableOpConE/Conv2D/ReadVariableOp2<
ConE2/BiasAdd/ReadVariableOpConE2/BiasAdd/ReadVariableOp2:
ConE2/Conv2D/ReadVariableOpConE2/Conv2D/ReadVariableOp2<
ConE3/BiasAdd/ReadVariableOpConE3/BiasAdd/ReadVariableOp2:
ConE3/Conv2D/ReadVariableOpConE3/Conv2D/ReadVariableOp2<
ConE4/BiasAdd/ReadVariableOpConE4/BiasAdd/ReadVariableOp2:
ConE4/Conv2D/ReadVariableOpConE4/Conv2D/ReadVariableOp2:
Hid1/BiasAdd/ReadVariableOpHid1/BiasAdd/ReadVariableOp28
Hid1/MatMul/ReadVariableOpHid1/MatMul/ReadVariableOp2:
Hid2/BiasAdd/ReadVariableOpHid2/BiasAdd/ReadVariableOp28
Hid2/MatMul/ReadVariableOpHid2/MatMul/ReadVariableOp2:
HidA/BiasAdd/ReadVariableOpHidA/BiasAdd/ReadVariableOp28
HidA/MatMul/ReadVariableOpHidA/MatMul/ReadVariableOp2<
HidA2/BiasAdd/ReadVariableOpHidA2/BiasAdd/ReadVariableOp2:
HidA2/MatMul/ReadVariableOpHidA2/MatMul/ReadVariableOp2<
HidA3/BiasAdd/ReadVariableOpHidA3/BiasAdd/ReadVariableOp2:
HidA3/MatMul/ReadVariableOpHidA3/MatMul/ReadVariableOp2<
HidA4/BiasAdd/ReadVariableOpHidA4/BiasAdd/ReadVariableOp2:
HidA4/MatMul/ReadVariableOpHidA4/MatMul/ReadVariableOp2:
HidC/BiasAdd/ReadVariableOpHidC/BiasAdd/ReadVariableOp28
HidC/MatMul/ReadVariableOpHidC/MatMul/ReadVariableOp2<
HidC2/BiasAdd/ReadVariableOpHidC2/BiasAdd/ReadVariableOp2:
HidC2/MatMul/ReadVariableOpHidC2/MatMul/ReadVariableOp2<
HidC3/BiasAdd/ReadVariableOpHidC3/BiasAdd/ReadVariableOp2:
HidC3/MatMul/ReadVariableOpHidC3/MatMul/ReadVariableOp2<
HidC4/BiasAdd/ReadVariableOpHidC4/BiasAdd/ReadVariableOp2:
HidC4/MatMul/ReadVariableOpHidC4/MatMul/ReadVariableOp2:
HidE/BiasAdd/ReadVariableOpHidE/BiasAdd/ReadVariableOp28
HidE/MatMul/ReadVariableOpHidE/MatMul/ReadVariableOp2<
HidE2/BiasAdd/ReadVariableOpHidE2/BiasAdd/ReadVariableOp2:
HidE2/MatMul/ReadVariableOpHidE2/MatMul/ReadVariableOp2<
HidE3/BiasAdd/ReadVariableOpHidE3/BiasAdd/ReadVariableOp2:
HidE3/MatMul/ReadVariableOpHidE3/MatMul/ReadVariableOp2<
HidE4/BiasAdd/ReadVariableOpHidE4/BiasAdd/ReadVariableOp2:
HidE4/MatMul/ReadVariableOpHidE4/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_4_layer_call_fn_230516

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_2269532
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_PoolE2_layer_call_and_return_conditional_losses_230239

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_concatenate_layer_call_and_return_conditional_losses_227279

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs:P	L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:P
L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
A__inference_HidE4_layer_call_and_return_conditional_losses_231099

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
C
'__inference_PoolC2_layer_call_fn_230214

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
:?????????"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolC2_layer_call_and_return_conditional_losses_2267532
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????D;:W S
/
_output_shapes
:?????????D;
 
_user_specified_nameinputs
?
?
&__inference_ConE3_layer_call_fn_230293

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConE3_layer_call_and_return_conditional_losses_2267722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
^
B__inference_PoolC2_layer_call_and_return_conditional_losses_230224

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????"*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????"2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????D;:W S
/
_output_shapes
:?????????D;
 
_user_specified_nameinputs
?
c
*__inference_dropout_1_layer_call_fn_230080

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F=$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_2280682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????F=$2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????F=$22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????F=$
 
_user_specified_nameinputs
??
?h
"__inference__traced_restore_232251
file_prefix6
assignvariableop_cona_kernel:*
assignvariableop_1_cona_bias:8
assignvariableop_2_conc_kernel:	$*
assignvariableop_3_conc_bias:$8
assignvariableop_4_cone_kernel:0*
assignvariableop_5_cone_bias:09
assignvariableop_6_cona2_kernel:+
assignvariableop_7_cona2_bias:9
assignvariableop_8_conc2_kernel:$+
assignvariableop_9_conc2_bias::
 assignvariableop_10_cone2_kernel:0,
assignvariableop_11_cone2_bias::
 assignvariableop_12_cona3_kernel:,
assignvariableop_13_cona3_bias::
 assignvariableop_14_conc3_kernel:,
assignvariableop_15_conc3_bias::
 assignvariableop_16_cone3_kernel:,
assignvariableop_17_cone3_bias::
 assignvariableop_18_cona4_kernel:,
assignvariableop_19_cona4_bias::
 assignvariableop_20_conc4_kernel:,
assignvariableop_21_conc4_bias::
 assignvariableop_22_cone4_kernel:,
assignvariableop_23_cone4_bias:4
assignvariableop_24_hida_kernel:???,
assignvariableop_25_hida_bias:	?5
 assignvariableop_26_hida2_kernel:???-
assignvariableop_27_hida2_bias:	?4
 assignvariableop_28_hida3_kernel:
?#?-
assignvariableop_29_hida3_bias:	?4
 assignvariableop_30_hida4_kernel:
??-
assignvariableop_31_hida4_bias:	?4
assignvariableop_32_hidc_kernel:??	?,
assignvariableop_33_hidc_bias:	?5
 assignvariableop_34_hidc2_kernel:???-
assignvariableop_35_hidc2_bias:	?4
 assignvariableop_36_hidc3_kernel:
? ?-
assignvariableop_37_hidc3_bias:	?4
 assignvariableop_38_hidc4_kernel:
??-
assignvariableop_39_hidc4_bias:	?4
assignvariableop_40_hide_kernel:???,
assignvariableop_41_hide_bias:	?5
 assignvariableop_42_hide2_kernel:???-
assignvariableop_43_hide2_bias:	?4
 assignvariableop_44_hide3_kernel:
??-
assignvariableop_45_hide3_bias:	?4
 assignvariableop_46_hide4_kernel:
??-
assignvariableop_47_hide4_bias:	?3
assignvariableop_48_hid1_kernel:
??,
assignvariableop_49_hid1_bias:	?3
assignvariableop_50_hid2_kernel:
??,
assignvariableop_51_hid2_bias:	?4
!assignvariableop_52_output_kernel:	?(-
assignvariableop_53_output_bias:('
assignvariableop_54_adam_iter:	 )
assignvariableop_55_adam_beta_1: )
assignvariableop_56_adam_beta_2: (
assignvariableop_57_adam_decay: 0
&assignvariableop_58_adam_learning_rate: #
assignvariableop_59_total: #
assignvariableop_60_count: %
assignvariableop_61_total_1: %
assignvariableop_62_count_1: @
&assignvariableop_63_adam_cona_kernel_m:2
$assignvariableop_64_adam_cona_bias_m:@
&assignvariableop_65_adam_conc_kernel_m:	$2
$assignvariableop_66_adam_conc_bias_m:$@
&assignvariableop_67_adam_cone_kernel_m:02
$assignvariableop_68_adam_cone_bias_m:0A
'assignvariableop_69_adam_cona2_kernel_m:3
%assignvariableop_70_adam_cona2_bias_m:A
'assignvariableop_71_adam_conc2_kernel_m:$3
%assignvariableop_72_adam_conc2_bias_m:A
'assignvariableop_73_adam_cone2_kernel_m:03
%assignvariableop_74_adam_cone2_bias_m:A
'assignvariableop_75_adam_cona3_kernel_m:3
%assignvariableop_76_adam_cona3_bias_m:A
'assignvariableop_77_adam_conc3_kernel_m:3
%assignvariableop_78_adam_conc3_bias_m:A
'assignvariableop_79_adam_cone3_kernel_m:3
%assignvariableop_80_adam_cone3_bias_m:A
'assignvariableop_81_adam_cona4_kernel_m:3
%assignvariableop_82_adam_cona4_bias_m:A
'assignvariableop_83_adam_conc4_kernel_m:3
%assignvariableop_84_adam_conc4_bias_m:A
'assignvariableop_85_adam_cone4_kernel_m:3
%assignvariableop_86_adam_cone4_bias_m:;
&assignvariableop_87_adam_hida_kernel_m:???3
$assignvariableop_88_adam_hida_bias_m:	?<
'assignvariableop_89_adam_hida2_kernel_m:???4
%assignvariableop_90_adam_hida2_bias_m:	?;
'assignvariableop_91_adam_hida3_kernel_m:
?#?4
%assignvariableop_92_adam_hida3_bias_m:	?;
'assignvariableop_93_adam_hida4_kernel_m:
??4
%assignvariableop_94_adam_hida4_bias_m:	?;
&assignvariableop_95_adam_hidc_kernel_m:??	?3
$assignvariableop_96_adam_hidc_bias_m:	?<
'assignvariableop_97_adam_hidc2_kernel_m:???4
%assignvariableop_98_adam_hidc2_bias_m:	?;
'assignvariableop_99_adam_hidc3_kernel_m:
? ?5
&assignvariableop_100_adam_hidc3_bias_m:	?<
(assignvariableop_101_adam_hidc4_kernel_m:
??5
&assignvariableop_102_adam_hidc4_bias_m:	?<
'assignvariableop_103_adam_hide_kernel_m:???4
%assignvariableop_104_adam_hide_bias_m:	?=
(assignvariableop_105_adam_hide2_kernel_m:???5
&assignvariableop_106_adam_hide2_bias_m:	?<
(assignvariableop_107_adam_hide3_kernel_m:
??5
&assignvariableop_108_adam_hide3_bias_m:	?<
(assignvariableop_109_adam_hide4_kernel_m:
??5
&assignvariableop_110_adam_hide4_bias_m:	?;
'assignvariableop_111_adam_hid1_kernel_m:
??4
%assignvariableop_112_adam_hid1_bias_m:	?;
'assignvariableop_113_adam_hid2_kernel_m:
??4
%assignvariableop_114_adam_hid2_bias_m:	?<
)assignvariableop_115_adam_output_kernel_m:	?(5
'assignvariableop_116_adam_output_bias_m:(A
'assignvariableop_117_adam_cona_kernel_v:3
%assignvariableop_118_adam_cona_bias_v:A
'assignvariableop_119_adam_conc_kernel_v:	$3
%assignvariableop_120_adam_conc_bias_v:$A
'assignvariableop_121_adam_cone_kernel_v:03
%assignvariableop_122_adam_cone_bias_v:0B
(assignvariableop_123_adam_cona2_kernel_v:4
&assignvariableop_124_adam_cona2_bias_v:B
(assignvariableop_125_adam_conc2_kernel_v:$4
&assignvariableop_126_adam_conc2_bias_v:B
(assignvariableop_127_adam_cone2_kernel_v:04
&assignvariableop_128_adam_cone2_bias_v:B
(assignvariableop_129_adam_cona3_kernel_v:4
&assignvariableop_130_adam_cona3_bias_v:B
(assignvariableop_131_adam_conc3_kernel_v:4
&assignvariableop_132_adam_conc3_bias_v:B
(assignvariableop_133_adam_cone3_kernel_v:4
&assignvariableop_134_adam_cone3_bias_v:B
(assignvariableop_135_adam_cona4_kernel_v:4
&assignvariableop_136_adam_cona4_bias_v:B
(assignvariableop_137_adam_conc4_kernel_v:4
&assignvariableop_138_adam_conc4_bias_v:B
(assignvariableop_139_adam_cone4_kernel_v:4
&assignvariableop_140_adam_cone4_bias_v:<
'assignvariableop_141_adam_hida_kernel_v:???4
%assignvariableop_142_adam_hida_bias_v:	?=
(assignvariableop_143_adam_hida2_kernel_v:???5
&assignvariableop_144_adam_hida2_bias_v:	?<
(assignvariableop_145_adam_hida3_kernel_v:
?#?5
&assignvariableop_146_adam_hida3_bias_v:	?<
(assignvariableop_147_adam_hida4_kernel_v:
??5
&assignvariableop_148_adam_hida4_bias_v:	?<
'assignvariableop_149_adam_hidc_kernel_v:??	?4
%assignvariableop_150_adam_hidc_bias_v:	?=
(assignvariableop_151_adam_hidc2_kernel_v:???5
&assignvariableop_152_adam_hidc2_bias_v:	?<
(assignvariableop_153_adam_hidc3_kernel_v:
? ?5
&assignvariableop_154_adam_hidc3_bias_v:	?<
(assignvariableop_155_adam_hidc4_kernel_v:
??5
&assignvariableop_156_adam_hidc4_bias_v:	?<
'assignvariableop_157_adam_hide_kernel_v:???4
%assignvariableop_158_adam_hide_bias_v:	?=
(assignvariableop_159_adam_hide2_kernel_v:???5
&assignvariableop_160_adam_hide2_bias_v:	?<
(assignvariableop_161_adam_hide3_kernel_v:
??5
&assignvariableop_162_adam_hide3_bias_v:	?<
(assignvariableop_163_adam_hide4_kernel_v:
??5
&assignvariableop_164_adam_hide4_bias_v:	?;
'assignvariableop_165_adam_hid1_kernel_v:
??4
%assignvariableop_166_adam_hid1_bias_v:	?;
'assignvariableop_167_adam_hid2_kernel_v:
??4
%assignvariableop_168_adam_hid2_bias_v:	?<
)assignvariableop_169_adam_output_kernel_v:	?(5
'assignvariableop_170_adam_output_bias_v:(
identity_172??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_120?AssignVariableOp_121?AssignVariableOp_122?AssignVariableOp_123?AssignVariableOp_124?AssignVariableOp_125?AssignVariableOp_126?AssignVariableOp_127?AssignVariableOp_128?AssignVariableOp_129?AssignVariableOp_13?AssignVariableOp_130?AssignVariableOp_131?AssignVariableOp_132?AssignVariableOp_133?AssignVariableOp_134?AssignVariableOp_135?AssignVariableOp_136?AssignVariableOp_137?AssignVariableOp_138?AssignVariableOp_139?AssignVariableOp_14?AssignVariableOp_140?AssignVariableOp_141?AssignVariableOp_142?AssignVariableOp_143?AssignVariableOp_144?AssignVariableOp_145?AssignVariableOp_146?AssignVariableOp_147?AssignVariableOp_148?AssignVariableOp_149?AssignVariableOp_15?AssignVariableOp_150?AssignVariableOp_151?AssignVariableOp_152?AssignVariableOp_153?AssignVariableOp_154?AssignVariableOp_155?AssignVariableOp_156?AssignVariableOp_157?AssignVariableOp_158?AssignVariableOp_159?AssignVariableOp_16?AssignVariableOp_160?AssignVariableOp_161?AssignVariableOp_162?AssignVariableOp_163?AssignVariableOp_164?AssignVariableOp_165?AssignVariableOp_166?AssignVariableOp_167?AssignVariableOp_168?AssignVariableOp_169?AssignVariableOp_17?AssignVariableOp_170?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?b
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?a
value?aB?a?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-26/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-26/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-26/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-26/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
?2?	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_cona_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_cona_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_conc_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_conc_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_cone_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_cone_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_cona2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_cona2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_conc2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_conc2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp assignvariableop_10_cone2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_cone2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp assignvariableop_12_cona3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_cona3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp assignvariableop_14_conc3_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_conc3_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp assignvariableop_16_cone3_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_cone3_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp assignvariableop_18_cona4_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_cona4_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp assignvariableop_20_conc4_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_conc4_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp assignvariableop_22_cone4_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_cone4_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_hida_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_hida_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp assignvariableop_26_hida2_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_hida2_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp assignvariableop_28_hida3_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_hida3_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp assignvariableop_30_hida4_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_hida4_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_hidc_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_hidc_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp assignvariableop_34_hidc2_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpassignvariableop_35_hidc2_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp assignvariableop_36_hidc3_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpassignvariableop_37_hidc3_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp assignvariableop_38_hidc4_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpassignvariableop_39_hidc4_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpassignvariableop_40_hide_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpassignvariableop_41_hide_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp assignvariableop_42_hide2_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpassignvariableop_43_hide2_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp assignvariableop_44_hide3_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpassignvariableop_45_hide3_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp assignvariableop_46_hide4_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOpassignvariableop_47_hide4_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOpassignvariableop_48_hid1_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOpassignvariableop_49_hid1_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpassignvariableop_50_hid2_kernelIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOpassignvariableop_51_hid2_biasIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp!assignvariableop_52_output_kernelIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOpassignvariableop_53_output_biasIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOpassignvariableop_54_adam_iterIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOpassignvariableop_55_adam_beta_1Identity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOpassignvariableop_56_adam_beta_2Identity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOpassignvariableop_57_adam_decayIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp&assignvariableop_58_adam_learning_rateIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOpassignvariableop_59_totalIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOpassignvariableop_60_countIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOpassignvariableop_61_total_1Identity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOpassignvariableop_62_count_1Identity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp&assignvariableop_63_adam_cona_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp$assignvariableop_64_adam_cona_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp&assignvariableop_65_adam_conc_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp$assignvariableop_66_adam_conc_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp&assignvariableop_67_adam_cone_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp$assignvariableop_68_adam_cone_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp'assignvariableop_69_adam_cona2_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp%assignvariableop_70_adam_cona2_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp'assignvariableop_71_adam_conc2_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp%assignvariableop_72_adam_conc2_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp'assignvariableop_73_adam_cone2_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp%assignvariableop_74_adam_cone2_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp'assignvariableop_75_adam_cona3_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp%assignvariableop_76_adam_cona3_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp'assignvariableop_77_adam_conc3_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp%assignvariableop_78_adam_conc3_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp'assignvariableop_79_adam_cone3_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp%assignvariableop_80_adam_cone3_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp'assignvariableop_81_adam_cona4_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp%assignvariableop_82_adam_cona4_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp'assignvariableop_83_adam_conc4_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp%assignvariableop_84_adam_conc4_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp'assignvariableop_85_adam_cone4_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp%assignvariableop_86_adam_cone4_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp&assignvariableop_87_adam_hida_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp$assignvariableop_88_adam_hida_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp'assignvariableop_89_adam_hida2_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp%assignvariableop_90_adam_hida2_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp'assignvariableop_91_adam_hida3_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp%assignvariableop_92_adam_hida3_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp'assignvariableop_93_adam_hida4_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp%assignvariableop_94_adam_hida4_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp&assignvariableop_95_adam_hidc_kernel_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp$assignvariableop_96_adam_hidc_bias_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOp'assignvariableop_97_adam_hidc2_kernel_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOp%assignvariableop_98_adam_hidc2_bias_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOp'assignvariableop_99_adam_hidc3_kernel_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOp&assignvariableop_100_adam_hidc3_bias_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOp(assignvariableop_101_adam_hidc4_kernel_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOp&assignvariableop_102_adam_hidc4_bias_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOp'assignvariableop_103_adam_hide_kernel_mIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104?
AssignVariableOp_104AssignVariableOp%assignvariableop_104_adam_hide_bias_mIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105?
AssignVariableOp_105AssignVariableOp(assignvariableop_105_adam_hide2_kernel_mIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106?
AssignVariableOp_106AssignVariableOp&assignvariableop_106_adam_hide2_bias_mIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107?
AssignVariableOp_107AssignVariableOp(assignvariableop_107_adam_hide3_kernel_mIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108?
AssignVariableOp_108AssignVariableOp&assignvariableop_108_adam_hide3_bias_mIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109?
AssignVariableOp_109AssignVariableOp(assignvariableop_109_adam_hide4_kernel_mIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110?
AssignVariableOp_110AssignVariableOp&assignvariableop_110_adam_hide4_bias_mIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111?
AssignVariableOp_111AssignVariableOp'assignvariableop_111_adam_hid1_kernel_mIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112?
AssignVariableOp_112AssignVariableOp%assignvariableop_112_adam_hid1_bias_mIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113?
AssignVariableOp_113AssignVariableOp'assignvariableop_113_adam_hid2_kernel_mIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114?
AssignVariableOp_114AssignVariableOp%assignvariableop_114_adam_hid2_bias_mIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115?
AssignVariableOp_115AssignVariableOp)assignvariableop_115_adam_output_kernel_mIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116?
AssignVariableOp_116AssignVariableOp'assignvariableop_116_adam_output_bias_mIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117?
AssignVariableOp_117AssignVariableOp'assignvariableop_117_adam_cona_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118?
AssignVariableOp_118AssignVariableOp%assignvariableop_118_adam_cona_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119?
AssignVariableOp_119AssignVariableOp'assignvariableop_119_adam_conc_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120?
AssignVariableOp_120AssignVariableOp%assignvariableop_120_adam_conc_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121?
AssignVariableOp_121AssignVariableOp'assignvariableop_121_adam_cone_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122?
AssignVariableOp_122AssignVariableOp%assignvariableop_122_adam_cone_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123?
AssignVariableOp_123AssignVariableOp(assignvariableop_123_adam_cona2_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124?
AssignVariableOp_124AssignVariableOp&assignvariableop_124_adam_cona2_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125?
AssignVariableOp_125AssignVariableOp(assignvariableop_125_adam_conc2_kernel_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126?
AssignVariableOp_126AssignVariableOp&assignvariableop_126_adam_conc2_bias_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127?
AssignVariableOp_127AssignVariableOp(assignvariableop_127_adam_cone2_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128?
AssignVariableOp_128AssignVariableOp&assignvariableop_128_adam_cone2_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129?
AssignVariableOp_129AssignVariableOp(assignvariableop_129_adam_cona3_kernel_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130?
AssignVariableOp_130AssignVariableOp&assignvariableop_130_adam_cona3_bias_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131?
AssignVariableOp_131AssignVariableOp(assignvariableop_131_adam_conc3_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132?
AssignVariableOp_132AssignVariableOp&assignvariableop_132_adam_conc3_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_132q
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:2
Identity_133?
AssignVariableOp_133AssignVariableOp(assignvariableop_133_adam_cone3_kernel_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_133q
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:2
Identity_134?
AssignVariableOp_134AssignVariableOp&assignvariableop_134_adam_cone3_bias_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_134q
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:2
Identity_135?
AssignVariableOp_135AssignVariableOp(assignvariableop_135_adam_cona4_kernel_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_135q
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:2
Identity_136?
AssignVariableOp_136AssignVariableOp&assignvariableop_136_adam_cona4_bias_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_136q
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:2
Identity_137?
AssignVariableOp_137AssignVariableOp(assignvariableop_137_adam_conc4_kernel_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_137q
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:2
Identity_138?
AssignVariableOp_138AssignVariableOp&assignvariableop_138_adam_conc4_bias_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_138q
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:2
Identity_139?
AssignVariableOp_139AssignVariableOp(assignvariableop_139_adam_cone4_kernel_vIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139q
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:2
Identity_140?
AssignVariableOp_140AssignVariableOp&assignvariableop_140_adam_cone4_bias_vIdentity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_140q
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:2
Identity_141?
AssignVariableOp_141AssignVariableOp'assignvariableop_141_adam_hida_kernel_vIdentity_141:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_141q
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:2
Identity_142?
AssignVariableOp_142AssignVariableOp%assignvariableop_142_adam_hida_bias_vIdentity_142:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_142q
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:2
Identity_143?
AssignVariableOp_143AssignVariableOp(assignvariableop_143_adam_hida2_kernel_vIdentity_143:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_143q
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:2
Identity_144?
AssignVariableOp_144AssignVariableOp&assignvariableop_144_adam_hida2_bias_vIdentity_144:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_144q
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
:2
Identity_145?
AssignVariableOp_145AssignVariableOp(assignvariableop_145_adam_hida3_kernel_vIdentity_145:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_145q
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:2
Identity_146?
AssignVariableOp_146AssignVariableOp&assignvariableop_146_adam_hida3_bias_vIdentity_146:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_146q
Identity_147IdentityRestoreV2:tensors:147"/device:CPU:0*
T0*
_output_shapes
:2
Identity_147?
AssignVariableOp_147AssignVariableOp(assignvariableop_147_adam_hida4_kernel_vIdentity_147:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_147q
Identity_148IdentityRestoreV2:tensors:148"/device:CPU:0*
T0*
_output_shapes
:2
Identity_148?
AssignVariableOp_148AssignVariableOp&assignvariableop_148_adam_hida4_bias_vIdentity_148:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_148q
Identity_149IdentityRestoreV2:tensors:149"/device:CPU:0*
T0*
_output_shapes
:2
Identity_149?
AssignVariableOp_149AssignVariableOp'assignvariableop_149_adam_hidc_kernel_vIdentity_149:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_149q
Identity_150IdentityRestoreV2:tensors:150"/device:CPU:0*
T0*
_output_shapes
:2
Identity_150?
AssignVariableOp_150AssignVariableOp%assignvariableop_150_adam_hidc_bias_vIdentity_150:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_150q
Identity_151IdentityRestoreV2:tensors:151"/device:CPU:0*
T0*
_output_shapes
:2
Identity_151?
AssignVariableOp_151AssignVariableOp(assignvariableop_151_adam_hidc2_kernel_vIdentity_151:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_151q
Identity_152IdentityRestoreV2:tensors:152"/device:CPU:0*
T0*
_output_shapes
:2
Identity_152?
AssignVariableOp_152AssignVariableOp&assignvariableop_152_adam_hidc2_bias_vIdentity_152:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_152q
Identity_153IdentityRestoreV2:tensors:153"/device:CPU:0*
T0*
_output_shapes
:2
Identity_153?
AssignVariableOp_153AssignVariableOp(assignvariableop_153_adam_hidc3_kernel_vIdentity_153:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_153q
Identity_154IdentityRestoreV2:tensors:154"/device:CPU:0*
T0*
_output_shapes
:2
Identity_154?
AssignVariableOp_154AssignVariableOp&assignvariableop_154_adam_hidc3_bias_vIdentity_154:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_154q
Identity_155IdentityRestoreV2:tensors:155"/device:CPU:0*
T0*
_output_shapes
:2
Identity_155?
AssignVariableOp_155AssignVariableOp(assignvariableop_155_adam_hidc4_kernel_vIdentity_155:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_155q
Identity_156IdentityRestoreV2:tensors:156"/device:CPU:0*
T0*
_output_shapes
:2
Identity_156?
AssignVariableOp_156AssignVariableOp&assignvariableop_156_adam_hidc4_bias_vIdentity_156:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_156q
Identity_157IdentityRestoreV2:tensors:157"/device:CPU:0*
T0*
_output_shapes
:2
Identity_157?
AssignVariableOp_157AssignVariableOp'assignvariableop_157_adam_hide_kernel_vIdentity_157:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_157q
Identity_158IdentityRestoreV2:tensors:158"/device:CPU:0*
T0*
_output_shapes
:2
Identity_158?
AssignVariableOp_158AssignVariableOp%assignvariableop_158_adam_hide_bias_vIdentity_158:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_158q
Identity_159IdentityRestoreV2:tensors:159"/device:CPU:0*
T0*
_output_shapes
:2
Identity_159?
AssignVariableOp_159AssignVariableOp(assignvariableop_159_adam_hide2_kernel_vIdentity_159:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_159q
Identity_160IdentityRestoreV2:tensors:160"/device:CPU:0*
T0*
_output_shapes
:2
Identity_160?
AssignVariableOp_160AssignVariableOp&assignvariableop_160_adam_hide2_bias_vIdentity_160:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_160q
Identity_161IdentityRestoreV2:tensors:161"/device:CPU:0*
T0*
_output_shapes
:2
Identity_161?
AssignVariableOp_161AssignVariableOp(assignvariableop_161_adam_hide3_kernel_vIdentity_161:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_161q
Identity_162IdentityRestoreV2:tensors:162"/device:CPU:0*
T0*
_output_shapes
:2
Identity_162?
AssignVariableOp_162AssignVariableOp&assignvariableop_162_adam_hide3_bias_vIdentity_162:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_162q
Identity_163IdentityRestoreV2:tensors:163"/device:CPU:0*
T0*
_output_shapes
:2
Identity_163?
AssignVariableOp_163AssignVariableOp(assignvariableop_163_adam_hide4_kernel_vIdentity_163:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_163q
Identity_164IdentityRestoreV2:tensors:164"/device:CPU:0*
T0*
_output_shapes
:2
Identity_164?
AssignVariableOp_164AssignVariableOp&assignvariableop_164_adam_hide4_bias_vIdentity_164:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_164q
Identity_165IdentityRestoreV2:tensors:165"/device:CPU:0*
T0*
_output_shapes
:2
Identity_165?
AssignVariableOp_165AssignVariableOp'assignvariableop_165_adam_hid1_kernel_vIdentity_165:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_165q
Identity_166IdentityRestoreV2:tensors:166"/device:CPU:0*
T0*
_output_shapes
:2
Identity_166?
AssignVariableOp_166AssignVariableOp%assignvariableop_166_adam_hid1_bias_vIdentity_166:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_166q
Identity_167IdentityRestoreV2:tensors:167"/device:CPU:0*
T0*
_output_shapes
:2
Identity_167?
AssignVariableOp_167AssignVariableOp'assignvariableop_167_adam_hid2_kernel_vIdentity_167:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_167q
Identity_168IdentityRestoreV2:tensors:168"/device:CPU:0*
T0*
_output_shapes
:2
Identity_168?
AssignVariableOp_168AssignVariableOp%assignvariableop_168_adam_hid2_bias_vIdentity_168:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_168q
Identity_169IdentityRestoreV2:tensors:169"/device:CPU:0*
T0*
_output_shapes
:2
Identity_169?
AssignVariableOp_169AssignVariableOp)assignvariableop_169_adam_output_kernel_vIdentity_169:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_169q
Identity_170IdentityRestoreV2:tensors:170"/device:CPU:0*
T0*
_output_shapes
:2
Identity_170?
AssignVariableOp_170AssignVariableOp'assignvariableop_170_adam_output_bias_vIdentity_170:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1709
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_171Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_165^AssignVariableOp_166^AssignVariableOp_167^AssignVariableOp_168^AssignVariableOp_169^AssignVariableOp_17^AssignVariableOp_170^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_171i
Identity_172IdentityIdentity_171:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_172?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_165^AssignVariableOp_166^AssignVariableOp_167^AssignVariableOp_168^AssignVariableOp_169^AssignVariableOp_17^AssignVariableOp_170^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"%
identity_172Identity_172:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382,
AssignVariableOp_139AssignVariableOp_1392*
AssignVariableOp_14AssignVariableOp_142,
AssignVariableOp_140AssignVariableOp_1402,
AssignVariableOp_141AssignVariableOp_1412,
AssignVariableOp_142AssignVariableOp_1422,
AssignVariableOp_143AssignVariableOp_1432,
AssignVariableOp_144AssignVariableOp_1442,
AssignVariableOp_145AssignVariableOp_1452,
AssignVariableOp_146AssignVariableOp_1462,
AssignVariableOp_147AssignVariableOp_1472,
AssignVariableOp_148AssignVariableOp_1482,
AssignVariableOp_149AssignVariableOp_1492*
AssignVariableOp_15AssignVariableOp_152,
AssignVariableOp_150AssignVariableOp_1502,
AssignVariableOp_151AssignVariableOp_1512,
AssignVariableOp_152AssignVariableOp_1522,
AssignVariableOp_153AssignVariableOp_1532,
AssignVariableOp_154AssignVariableOp_1542,
AssignVariableOp_155AssignVariableOp_1552,
AssignVariableOp_156AssignVariableOp_1562,
AssignVariableOp_157AssignVariableOp_1572,
AssignVariableOp_158AssignVariableOp_1582,
AssignVariableOp_159AssignVariableOp_1592*
AssignVariableOp_16AssignVariableOp_162,
AssignVariableOp_160AssignVariableOp_1602,
AssignVariableOp_161AssignVariableOp_1612,
AssignVariableOp_162AssignVariableOp_1622,
AssignVariableOp_163AssignVariableOp_1632,
AssignVariableOp_164AssignVariableOp_1642,
AssignVariableOp_165AssignVariableOp_1652,
AssignVariableOp_166AssignVariableOp_1662,
AssignVariableOp_167AssignVariableOp_1672,
AssignVariableOp_168AssignVariableOp_1682,
AssignVariableOp_169AssignVariableOp_1692*
AssignVariableOp_17AssignVariableOp_172,
AssignVariableOp_170AssignVariableOp_1702*
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
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
&__inference_model_layer_call_fn_228654	
input!
unknown:0
	unknown_0:0#
	unknown_1:	$
	unknown_2:$#
	unknown_3:
	unknown_4:#
	unknown_5:0
	unknown_6:#
	unknown_7:$
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:$

unknown_15:

unknown_16:$

unknown_17:

unknown_18:$

unknown_19:

unknown_20:$

unknown_21:

unknown_22:

unknown_23:???

unknown_24:	?

unknown_25:???

unknown_26:	?

unknown_27:
?#?

unknown_28:	?

unknown_29:
??

unknown_30:	?

unknown_31:??	?

unknown_32:	?

unknown_33:???

unknown_34:	?

unknown_35:
? ?

unknown_36:	?

unknown_37:
??

unknown_38:	?

unknown_39:???

unknown_40:	?

unknown_41:???

unknown_42:	?

unknown_43:
??

unknown_44:	?

unknown_45:
??

unknown_46:	?

unknown_47:
??

unknown_48:	?

unknown_49:
??

unknown_50:	?

unknown_51:	?(

unknown_52:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2284302
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
^
B__inference_PoolA3_layer_call_and_return_conditional_losses_230324

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????!:W S
/
_output_shapes
:?????????!
 
_user_specified_nameinputs
?
F
*__inference_flatten_3_layer_call_fn_230765

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
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_2270322
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_PoolA2_layer_call_and_return_conditional_losses_226406

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_3_layer_call_fn_230489

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
:?????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2269602
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????#:W S
/
_output_shapes
:?????????#
 
_user_specified_nameinputs
??
?
A__inference_model_layer_call_and_return_conditional_losses_227333

inputs%
cone_226614:0
cone_226616:0%
conc_226631:	$
conc_226633:$%
cona_226648:
cona_226650:&
cone2_226704:0
cone2_226706:&
conc2_226721:$
conc2_226723:&
cona2_226738:
cona2_226740:&
cone3_226773:
cone3_226775:&
conc3_226790:
conc3_226792:&
cona3_226807:
cona3_226809:&
cone4_226842:
cone4_226844:&
conc4_226859:
conc4_226861:&
cona4_226876:
cona4_226878: 
hida_227070:???
hida_227072:	?!
hida2_227087:???
hida2_227089:	? 
hida3_227104:
?#?
hida3_227106:	? 
hida4_227121:
??
hida4_227123:	? 
hidc_227138:??	?
hidc_227140:	?!
hidc2_227155:???
hidc2_227157:	? 
hidc3_227172:
? ?
hidc3_227174:	? 
hidc4_227189:
??
hidc4_227191:	? 
hide_227206:???
hide_227208:	?!
hide2_227223:???
hide2_227225:	? 
hide3_227240:
??
hide3_227242:	? 
hide4_227257:
??
hide4_227259:	?
hid1_227293:
??
hid1_227295:	?
hid2_227310:
??
hid2_227312:	? 
output_227327:	?(
output_227329:(
identity??ConA/StatefulPartitionedCall?ConA2/StatefulPartitionedCall?ConA3/StatefulPartitionedCall?ConA4/StatefulPartitionedCall?ConC/StatefulPartitionedCall?ConC2/StatefulPartitionedCall?ConC3/StatefulPartitionedCall?ConC4/StatefulPartitionedCall?ConE/StatefulPartitionedCall?ConE2/StatefulPartitionedCall?ConE3/StatefulPartitionedCall?ConE4/StatefulPartitionedCall?Hid1/StatefulPartitionedCall?Hid2/StatefulPartitionedCall?HidA/StatefulPartitionedCall?HidA2/StatefulPartitionedCall?HidA3/StatefulPartitionedCall?HidA4/StatefulPartitionedCall?HidC/StatefulPartitionedCall?HidC2/StatefulPartitionedCall?HidC3/StatefulPartitionedCall?HidC4/StatefulPartitionedCall?HidE/StatefulPartitionedCall?HidE2/StatefulPartitionedCall?HidE3/StatefulPartitionedCall?HidE4/StatefulPartitionedCall?output/StatefulPartitionedCall?
ConE/StatefulPartitionedCallStatefulPartitionedCallinputscone_226614cone_226616*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConE_layer_call_and_return_conditional_losses_2266132
ConE/StatefulPartitionedCall?
ConC/StatefulPartitionedCallStatefulPartitionedCallinputsconc_226631conc_226633*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConC_layer_call_and_return_conditional_losses_2266302
ConC/StatefulPartitionedCall?
ConA/StatefulPartitionedCallStatefulPartitionedCallinputscona_226648cona_226650*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConA_layer_call_and_return_conditional_losses_2266472
ConA/StatefulPartitionedCall?
PoolE/PartitionedCallPartitionedCall%ConE/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????B:0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolE_layer_call_and_return_conditional_losses_2266572
PoolE/PartitionedCall?
PoolC/PartitionedCallPartitionedCall%ConC/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F=$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolC_layer_call_and_return_conditional_losses_2266632
PoolC/PartitionedCall?
PoolA/PartitionedCallPartitionedCall%ConA/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolA_layer_call_and_return_conditional_losses_2266692
PoolA/PartitionedCall?
dropout_2/PartitionedCallPartitionedCallPoolE/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????B:0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2266762
dropout_2/PartitionedCall?
dropout_1/PartitionedCallPartitionedCallPoolC/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F=$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_2266832
dropout_1/PartitionedCall?
dropout/PartitionedCallPartitionedCallPoolA/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_2266902
dropout/PartitionedCall?
ConE2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0cone2_226704cone2_226706*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@8*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConE2_layer_call_and_return_conditional_losses_2267032
ConE2/StatefulPartitionedCall?
ConC2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conc2_226721conc2_226723*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????D;*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConC2_layer_call_and_return_conditional_losses_2267202
ConC2/StatefulPartitionedCall?
ConA2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0cona2_226738cona2_226740*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????G=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConA2_layer_call_and_return_conditional_losses_2267372
ConA2/StatefulPartitionedCall?
PoolE2/PartitionedCallPartitionedCall&ConE2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolE2_layer_call_and_return_conditional_losses_2267472
PoolE2/PartitionedCall?
PoolC2/PartitionedCallPartitionedCall&ConC2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolC2_layer_call_and_return_conditional_losses_2267532
PoolC2/PartitionedCall?
PoolA2/PartitionedCallPartitionedCall&ConA2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA2_layer_call_and_return_conditional_losses_2267592
PoolA2/PartitionedCall?
ConE3/StatefulPartitionedCallStatefulPartitionedCallPoolE2/PartitionedCall:output:0cone3_226773cone3_226775*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConE3_layer_call_and_return_conditional_losses_2267722
ConE3/StatefulPartitionedCall?
ConC3/StatefulPartitionedCallStatefulPartitionedCallPoolC2/PartitionedCall:output:0conc3_226790conc3_226792*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConC3_layer_call_and_return_conditional_losses_2267892
ConC3/StatefulPartitionedCall?
ConA3/StatefulPartitionedCallStatefulPartitionedCallPoolA2/PartitionedCall:output:0cona3_226807cona3_226809*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConA3_layer_call_and_return_conditional_losses_2268062
ConA3/StatefulPartitionedCall?
PoolE3/PartitionedCallPartitionedCall&ConE3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolE3_layer_call_and_return_conditional_losses_2268162
PoolE3/PartitionedCall?
PoolC3/PartitionedCallPartitionedCall&ConC3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolC3_layer_call_and_return_conditional_losses_2268222
PoolC3/PartitionedCall?
PoolA3/PartitionedCallPartitionedCall&ConA3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA3_layer_call_and_return_conditional_losses_2268282
PoolA3/PartitionedCall?
ConE4/StatefulPartitionedCallStatefulPartitionedCallPoolE3/PartitionedCall:output:0cone4_226842cone4_226844*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConE4_layer_call_and_return_conditional_losses_2268412
ConE4/StatefulPartitionedCall?
ConC4/StatefulPartitionedCallStatefulPartitionedCallPoolC3/PartitionedCall:output:0conc4_226859conc4_226861*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConC4_layer_call_and_return_conditional_losses_2268582
ConC4/StatefulPartitionedCall?
ConA4/StatefulPartitionedCallStatefulPartitionedCallPoolA3/PartitionedCall:output:0cona4_226876cona4_226878*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConA4_layer_call_and_return_conditional_losses_2268752
ConA4/StatefulPartitionedCall?
PoolE4/PartitionedCallPartitionedCall&ConE4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolE4_layer_call_and_return_conditional_losses_2268852
PoolE4/PartitionedCall?
PoolC4/PartitionedCallPartitionedCall&ConC4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolC4_layer_call_and_return_conditional_losses_2268912
PoolC4/PartitionedCall?
PoolA4/PartitionedCallPartitionedCall&ConA4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA4_layer_call_and_return_conditional_losses_2268972
PoolA4/PartitionedCall?
dropout_11/PartitionedCallPartitionedCallPoolE4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_2269042
dropout_11/PartitionedCall?
dropout_10/PartitionedCallPartitionedCallPoolE3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_2269112
dropout_10/PartitionedCall?
dropout_9/PartitionedCallPartitionedCallPoolE2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_2269182
dropout_9/PartitionedCall?
dropout_8/PartitionedCallPartitionedCallPoolC4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_2269252
dropout_8/PartitionedCall?
dropout_7/PartitionedCallPartitionedCallPoolC3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_2269322
dropout_7/PartitionedCall?
dropout_6/PartitionedCallPartitionedCallPoolC2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_2269392
dropout_6/PartitionedCall?
dropout_5/PartitionedCallPartitionedCallPoolA4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_2269462
dropout_5/PartitionedCall?
dropout_4/PartitionedCallPartitionedCallPoolA3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_2269532
dropout_4/PartitionedCall?
dropout_3/PartitionedCallPartitionedCallPoolA2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2269602
dropout_3/PartitionedCall?
flatten_11/PartitionedCallPartitionedCall#dropout_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_11_layer_call_and_return_conditional_losses_2269682
flatten_11/PartitionedCall?
flatten_10/PartitionedCallPartitionedCall#dropout_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_10_layer_call_and_return_conditional_losses_2269762
flatten_10/PartitionedCall?
flatten_9/PartitionedCallPartitionedCall"dropout_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_9_layer_call_and_return_conditional_losses_2269842
flatten_9/PartitionedCall?
flatten_8/PartitionedCallPartitionedCallPoolE/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_2269922
flatten_8/PartitionedCall?
flatten_7/PartitionedCallPartitionedCall"dropout_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_2270002
flatten_7/PartitionedCall?
flatten_6/PartitionedCallPartitionedCall"dropout_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_2270082
flatten_6/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall"dropout_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_2270162
flatten_5/PartitionedCall?
flatten_4/PartitionedCallPartitionedCallPoolC/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_2270242
flatten_4/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall"dropout_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_2270322
flatten_3/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall"dropout_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_2270402
flatten_2/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall"dropout_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2270482
flatten_1/PartitionedCall?
flatten/PartitionedCallPartitionedCallPoolA/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2270562
flatten/PartitionedCall?
HidA/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0hida_227070hida_227072*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_HidA_layer_call_and_return_conditional_losses_2270692
HidA/StatefulPartitionedCall?
HidA2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0hida2_227087hida2_227089*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidA2_layer_call_and_return_conditional_losses_2270862
HidA2/StatefulPartitionedCall?
HidA3/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0hida3_227104hida3_227106*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidA3_layer_call_and_return_conditional_losses_2271032
HidA3/StatefulPartitionedCall?
HidA4/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0hida4_227121hida4_227123*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidA4_layer_call_and_return_conditional_losses_2271202
HidA4/StatefulPartitionedCall?
HidC/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0hidc_227138hidc_227140*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_HidC_layer_call_and_return_conditional_losses_2271372
HidC/StatefulPartitionedCall?
HidC2/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0hidc2_227155hidc2_227157*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidC2_layer_call_and_return_conditional_losses_2271542
HidC2/StatefulPartitionedCall?
HidC3/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0hidc3_227172hidc3_227174*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidC3_layer_call_and_return_conditional_losses_2271712
HidC3/StatefulPartitionedCall?
HidC4/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0hidc4_227189hidc4_227191*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidC4_layer_call_and_return_conditional_losses_2271882
HidC4/StatefulPartitionedCall?
HidE/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0hide_227206hide_227208*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_HidE_layer_call_and_return_conditional_losses_2272052
HidE/StatefulPartitionedCall?
HidE2/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0hide2_227223hide2_227225*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidE2_layer_call_and_return_conditional_losses_2272222
HidE2/StatefulPartitionedCall?
HidE3/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0hide3_227240hide3_227242*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidE3_layer_call_and_return_conditional_losses_2272392
HidE3/StatefulPartitionedCall?
HidE4/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0hide4_227257hide4_227259*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidE4_layer_call_and_return_conditional_losses_2272562
HidE4/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall%HidA/StatefulPartitionedCall:output:0&HidA2/StatefulPartitionedCall:output:0&HidA3/StatefulPartitionedCall:output:0&HidA4/StatefulPartitionedCall:output:0%HidC/StatefulPartitionedCall:output:0&HidC2/StatefulPartitionedCall:output:0&HidC3/StatefulPartitionedCall:output:0&HidC4/StatefulPartitionedCall:output:0%HidE/StatefulPartitionedCall:output:0&HidE2/StatefulPartitionedCall:output:0&HidE3/StatefulPartitionedCall:output:0&HidE4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_2272792
concatenate/PartitionedCall?
Hid1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0hid1_227293hid1_227295*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_Hid1_layer_call_and_return_conditional_losses_2272922
Hid1/StatefulPartitionedCall?
Hid2/StatefulPartitionedCallStatefulPartitionedCall%Hid1/StatefulPartitionedCall:output:0hid2_227310hid2_227312*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_Hid2_layer_call_and_return_conditional_losses_2273092
Hid2/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall%Hid2/StatefulPartitionedCall:output:0output_227327output_227329*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_2273262 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????(2

Identity?
NoOpNoOp^ConA/StatefulPartitionedCall^ConA2/StatefulPartitionedCall^ConA3/StatefulPartitionedCall^ConA4/StatefulPartitionedCall^ConC/StatefulPartitionedCall^ConC2/StatefulPartitionedCall^ConC3/StatefulPartitionedCall^ConC4/StatefulPartitionedCall^ConE/StatefulPartitionedCall^ConE2/StatefulPartitionedCall^ConE3/StatefulPartitionedCall^ConE4/StatefulPartitionedCall^Hid1/StatefulPartitionedCall^Hid2/StatefulPartitionedCall^HidA/StatefulPartitionedCall^HidA2/StatefulPartitionedCall^HidA3/StatefulPartitionedCall^HidA4/StatefulPartitionedCall^HidC/StatefulPartitionedCall^HidC2/StatefulPartitionedCall^HidC3/StatefulPartitionedCall^HidC4/StatefulPartitionedCall^HidE/StatefulPartitionedCall^HidE2/StatefulPartitionedCall^HidE3/StatefulPartitionedCall^HidE4/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
ConA/StatefulPartitionedCallConA/StatefulPartitionedCall2>
ConA2/StatefulPartitionedCallConA2/StatefulPartitionedCall2>
ConA3/StatefulPartitionedCallConA3/StatefulPartitionedCall2>
ConA4/StatefulPartitionedCallConA4/StatefulPartitionedCall2<
ConC/StatefulPartitionedCallConC/StatefulPartitionedCall2>
ConC2/StatefulPartitionedCallConC2/StatefulPartitionedCall2>
ConC3/StatefulPartitionedCallConC3/StatefulPartitionedCall2>
ConC4/StatefulPartitionedCallConC4/StatefulPartitionedCall2<
ConE/StatefulPartitionedCallConE/StatefulPartitionedCall2>
ConE2/StatefulPartitionedCallConE2/StatefulPartitionedCall2>
ConE3/StatefulPartitionedCallConE3/StatefulPartitionedCall2>
ConE4/StatefulPartitionedCallConE4/StatefulPartitionedCall2<
Hid1/StatefulPartitionedCallHid1/StatefulPartitionedCall2<
Hid2/StatefulPartitionedCallHid2/StatefulPartitionedCall2<
HidA/StatefulPartitionedCallHidA/StatefulPartitionedCall2>
HidA2/StatefulPartitionedCallHidA2/StatefulPartitionedCall2>
HidA3/StatefulPartitionedCallHidA3/StatefulPartitionedCall2>
HidA4/StatefulPartitionedCallHidA4/StatefulPartitionedCall2<
HidC/StatefulPartitionedCallHidC/StatefulPartitionedCall2>
HidC2/StatefulPartitionedCallHidC2/StatefulPartitionedCall2>
HidC3/StatefulPartitionedCallHidC3/StatefulPartitionedCall2>
HidC4/StatefulPartitionedCallHidC4/StatefulPartitionedCall2<
HidE/StatefulPartitionedCallHidE/StatefulPartitionedCall2>
HidE2/StatefulPartitionedCallHidE2/StatefulPartitionedCall2>
HidE3/StatefulPartitionedCallHidE3/StatefulPartitionedCall2>
HidE4/StatefulPartitionedCallHidE4/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
&__inference_ConE2_layer_call_fn_230173

inputs!
unknown:0
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@8*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConE2_layer_call_and_return_conditional_losses_2267032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@82

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????B:0: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????B:0
 
_user_specified_nameinputs
?
a
E__inference_flatten_5_layer_call_and_return_conditional_losses_227016

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????M  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????":W S
/
_output_shapes
:?????????"
 
_user_specified_nameinputs
?
G
+__inference_dropout_11_layer_call_fn_230705

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_2269042
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_HidE_layer_call_fn_231028

inputs
unknown:???
	unknown_0:	?
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
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_HidE_layer_call_and_return_conditional_losses_2272052
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
@__inference_HidA_layer_call_and_return_conditional_losses_227069

inputs3
matmul_readvariableop_resource:???.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
&__inference_ConC3_layer_call_fn_230273

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConC3_layer_call_and_return_conditional_losses_2267892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????": : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????"
 
_user_specified_nameinputs
?
F
*__inference_dropout_9_layer_call_fn_230651

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
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_2269182
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
^
B__inference_PoolC4_layer_call_and_return_conditional_losses_230459

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
@__inference_HidE_layer_call_and_return_conditional_losses_231039

inputs3
matmul_readvariableop_resource:???.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_230771

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????H  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_7_layer_call_and_return_conditional_losses_227000

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
C
'__inference_PoolC2_layer_call_fn_230209

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
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolC2_layer_call_and_return_conditional_losses_2264282
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_11_layer_call_and_return_conditional_losses_226904

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_PoolC3_layer_call_and_return_conditional_losses_226822

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
c
*__inference_dropout_3_layer_call_fn_230494

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2277032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????#2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????#22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????#
 
_user_specified_nameinputs
?
d
+__inference_dropout_11_layer_call_fn_230710

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_2278872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
@__inference_Hid1_layer_call_and_return_conditional_losses_227292

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
D
(__inference_flatten_layer_call_fn_230732

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2270562
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????I?:W S
/
_output_shapes
:?????????I?
 
_user_specified_nameinputs
?
]
A__inference_PoolC_layer_call_and_return_conditional_losses_226663

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????F=$*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????F=$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????$:Y U
1
_output_shapes
:???????????$
 
_user_specified_nameinputs
?
]
A__inference_PoolC_layer_call_and_return_conditional_losses_230018

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_HidA2_layer_call_and_return_conditional_losses_230899

inputs3
matmul_readvariableop_resource:???.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
F__inference_flatten_11_layer_call_and_return_conditional_losses_230859

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????X  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_6_layer_call_fn_230798

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
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_2270082
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
A__inference_ConC4_layer_call_and_return_conditional_losses_226858

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_7_layer_call_fn_230602

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_2277952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_4_layer_call_and_return_conditional_losses_227024

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????xX 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????	2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????F=$:W S
/
_output_shapes
:?????????F=$
 
_user_specified_nameinputs
?
a
E__inference_flatten_9_layer_call_and_return_conditional_losses_226984

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? F  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
@__inference_ConC_layer_call_and_return_conditional_losses_229963

inputs8
conv2d_readvariableop_resource:	$-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	$*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????$2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????$2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
D
(__inference_dropout_layer_call_fn_230048

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
:?????????I?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_2266902
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????I?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????I?:W S
/
_output_shapes
:?????????I?
 
_user_specified_nameinputs
?
b
F__inference_flatten_10_layer_call_and_return_conditional_losses_226976

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????<  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
A__inference_HidC2_layer_call_and_return_conditional_losses_227154

inputs3
matmul_readvariableop_resource:???.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
A__inference_HidC4_layer_call_and_return_conditional_losses_227188

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
A__inference_HidC2_layer_call_and_return_conditional_losses_230979

inputs3
matmul_readvariableop_resource:???.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
&__inference_ConA4_layer_call_fn_230373

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConA4_layer_call_and_return_conditional_losses_2268752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_4_layer_call_fn_230521

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_2277262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_8_layer_call_fn_230820

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_2269922
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????B:0:W S
/
_output_shapes
:?????????B:0
 
_user_specified_nameinputs
?
B
&__inference_PoolC_layer_call_fn_230008

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
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolC_layer_call_and_return_conditional_losses_2263622
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_ConE4_layer_call_and_return_conditional_losses_230424

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_230553

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_4_layer_call_and_return_conditional_losses_227726

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
@__inference_HidC_layer_call_and_return_conditional_losses_227137

inputs3
matmul_readvariableop_resource:??	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:??	?*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????	
 
_user_specified_nameinputs
?
?
&__inference_ConE4_layer_call_fn_230413

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConE4_layer_call_and_return_conditional_losses_2268412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_7_layer_call_fn_230597

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_2269322
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_1_layer_call_fn_230743

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2270482
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????#:W S
/
_output_shapes
:?????????#
 
_user_specified_nameinputs
?
?
&__inference_ConC2_layer_call_fn_230153

inputs!
unknown:$
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????D;*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConC2_layer_call_and_return_conditional_losses_2267202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????D;2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????F=$: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????F=$
 
_user_specified_nameinputs
?
C
'__inference_PoolC3_layer_call_fn_230329

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
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolC3_layer_call_and_return_conditional_losses_2264942
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
&__inference_HidA4_layer_call_fn_230928

inputs
unknown:
??
	unknown_0:	?
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
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_HidA4_layer_call_and_return_conditional_losses_2271202
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_9_layer_call_and_return_conditional_losses_226918

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
^
B__inference_PoolC2_layer_call_and_return_conditional_losses_230219

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_230499

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????#2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????#2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????#:W S
/
_output_shapes
:?????????#
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
A
input8
serving_default_input:0???????????:
output0
StatefulPartitionedCall:0?????????(tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
layer_with_weights-7
layer-17
layer_with_weights-8
layer-18
layer-19
layer-20
layer-21
layer_with_weights-9
layer-22
layer_with_weights-10
layer-23
layer_with_weights-11
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
layer-30
 layer-31
!layer-32
"layer-33
#layer-34
$layer-35
%layer-36
&layer-37
'layer-38
(layer-39
)layer-40
*layer-41
+layer-42
,layer-43
-layer-44
.layer-45
/layer-46
0layer-47
1layer-48
2layer_with_weights-12
2layer-49
3layer_with_weights-13
3layer-50
4layer_with_weights-14
4layer-51
5layer_with_weights-15
5layer-52
6layer_with_weights-16
6layer-53
7layer_with_weights-17
7layer-54
8layer_with_weights-18
8layer-55
9layer_with_weights-19
9layer-56
:layer_with_weights-20
:layer-57
;layer_with_weights-21
;layer-58
<layer_with_weights-22
<layer-59
=layer_with_weights-23
=layer-60
>layer-61
?layer_with_weights-24
?layer-62
@layer_with_weights-25
@layer-63
Alayer_with_weights-26
Alayer-64
B	optimizer
C
signatures
#D_self_saveable_object_factories
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_network
D
#I_self_saveable_object_factories"
_tf_keras_input_layer
?

Jkernel
Kbias
#L_self_saveable_object_factories
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Qkernel
Rbias
#S_self_saveable_object_factories
Ttrainable_variables
U	variables
Vregularization_losses
W	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Xkernel
Ybias
#Z_self_saveable_object_factories
[trainable_variables
\	variables
]regularization_losses
^	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#__self_saveable_object_factories
`trainable_variables
a	variables
bregularization_losses
c	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#d_self_saveable_object_factories
etrainable_variables
f	variables
gregularization_losses
h	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#i_self_saveable_object_factories
jtrainable_variables
k	variables
lregularization_losses
m	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#n_self_saveable_object_factories
otrainable_variables
p	variables
qregularization_losses
r	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#s_self_saveable_object_factories
ttrainable_variables
u	variables
vregularization_losses
w	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#x_self_saveable_object_factories
ytrainable_variables
z	variables
{regularization_losses
|	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

}kernel
~bias
#_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?	
	?iter
?beta_1
?beta_2

?decay
?learning_rateJm?Km?Qm?Rm?Xm?Ym?}m?~m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?Jv?Kv?Qv?Rv?Xv?Yv?}v?~v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
?
J0
K1
Q2
R3
X4
Y5
}6
~7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53"
trackable_list_wrapper
?
J0
K1
Q2
R3
X4
Y5
}6
~7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
Etrainable_variables
?metrics
F	variables
 ?layer_regularization_losses
Gregularization_losses
?layers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
%:#2ConA/kernel
:2	ConA/bias
 "
trackable_dict_wrapper
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
Mtrainable_variables
?metrics
N	variables
 ?layer_regularization_losses
Oregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#	$2ConC/kernel
:$2	ConC/bias
 "
trackable_dict_wrapper
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
Ttrainable_variables
?metrics
U	variables
 ?layer_regularization_losses
Vregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#02ConE/kernel
:02	ConE/bias
 "
trackable_dict_wrapper
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
[trainable_variables
?metrics
\	variables
 ?layer_regularization_losses
]regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
`trainable_variables
?metrics
a	variables
 ?layer_regularization_losses
bregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
etrainable_variables
?metrics
f	variables
 ?layer_regularization_losses
gregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
jtrainable_variables
?metrics
k	variables
 ?layer_regularization_losses
lregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
otrainable_variables
?metrics
p	variables
 ?layer_regularization_losses
qregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
ttrainable_variables
?metrics
u	variables
 ?layer_regularization_losses
vregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
ytrainable_variables
?metrics
z	variables
 ?layer_regularization_losses
{regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$2ConA2/kernel
:2
ConA2/bias
 "
trackable_dict_wrapper
.
}0
~1"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$$2ConC2/kernel
:2
ConC2/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$02ConE2/kernel
:2
ConE2/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$2ConA3/kernel
:2
ConA3/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$2ConC3/kernel
:2
ConC3/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$2ConE3/kernel
:2
ConE3/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$2ConA4/kernel
:2
ConA4/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$2ConC4/kernel
:2
ConC4/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$2ConE4/kernel
:2
ConE4/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :???2HidA/kernel
:?2	HidA/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:???2HidA2/kernel
:?2
HidA2/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :
?#?2HidA3/kernel
:?2
HidA3/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :
??2HidA4/kernel
:?2
HidA4/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :??	?2HidC/kernel
:?2	HidC/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:???2HidC2/kernel
:?2
HidC2/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :
? ?2HidC3/kernel
:?2
HidC3/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :
??2HidC4/kernel
:?2
HidC4/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :???2HidE/kernel
:?2	HidE/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:???2HidE2/kernel
:?2
HidE2/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :
??2HidE3/kernel
:?2
HidE3/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :
??2HidE4/kernel
:?2
HidE4/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
??2Hid1/kernel
:?2	Hid1/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
??2Hid2/kernel
:?2	Hid2/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :	?(2output/kernel
:(2output/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653
754
855
956
:57
;58
<59
=60
>61
?62
@63
A64"
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
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
*:(2Adam/ConA/kernel/m
:2Adam/ConA/bias/m
*:(	$2Adam/ConC/kernel/m
:$2Adam/ConC/bias/m
*:(02Adam/ConE/kernel/m
:02Adam/ConE/bias/m
+:)2Adam/ConA2/kernel/m
:2Adam/ConA2/bias/m
+:)$2Adam/ConC2/kernel/m
:2Adam/ConC2/bias/m
+:)02Adam/ConE2/kernel/m
:2Adam/ConE2/bias/m
+:)2Adam/ConA3/kernel/m
:2Adam/ConA3/bias/m
+:)2Adam/ConC3/kernel/m
:2Adam/ConC3/bias/m
+:)2Adam/ConE3/kernel/m
:2Adam/ConE3/bias/m
+:)2Adam/ConA4/kernel/m
:2Adam/ConA4/bias/m
+:)2Adam/ConC4/kernel/m
:2Adam/ConC4/bias/m
+:)2Adam/ConE4/kernel/m
:2Adam/ConE4/bias/m
%:#???2Adam/HidA/kernel/m
:?2Adam/HidA/bias/m
&:$???2Adam/HidA2/kernel/m
:?2Adam/HidA2/bias/m
%:#
?#?2Adam/HidA3/kernel/m
:?2Adam/HidA3/bias/m
%:#
??2Adam/HidA4/kernel/m
:?2Adam/HidA4/bias/m
%:#??	?2Adam/HidC/kernel/m
:?2Adam/HidC/bias/m
&:$???2Adam/HidC2/kernel/m
:?2Adam/HidC2/bias/m
%:#
? ?2Adam/HidC3/kernel/m
:?2Adam/HidC3/bias/m
%:#
??2Adam/HidC4/kernel/m
:?2Adam/HidC4/bias/m
%:#???2Adam/HidE/kernel/m
:?2Adam/HidE/bias/m
&:$???2Adam/HidE2/kernel/m
:?2Adam/HidE2/bias/m
%:#
??2Adam/HidE3/kernel/m
:?2Adam/HidE3/bias/m
%:#
??2Adam/HidE4/kernel/m
:?2Adam/HidE4/bias/m
$:"
??2Adam/Hid1/kernel/m
:?2Adam/Hid1/bias/m
$:"
??2Adam/Hid2/kernel/m
:?2Adam/Hid2/bias/m
%:#	?(2Adam/output/kernel/m
:(2Adam/output/bias/m
*:(2Adam/ConA/kernel/v
:2Adam/ConA/bias/v
*:(	$2Adam/ConC/kernel/v
:$2Adam/ConC/bias/v
*:(02Adam/ConE/kernel/v
:02Adam/ConE/bias/v
+:)2Adam/ConA2/kernel/v
:2Adam/ConA2/bias/v
+:)$2Adam/ConC2/kernel/v
:2Adam/ConC2/bias/v
+:)02Adam/ConE2/kernel/v
:2Adam/ConE2/bias/v
+:)2Adam/ConA3/kernel/v
:2Adam/ConA3/bias/v
+:)2Adam/ConC3/kernel/v
:2Adam/ConC3/bias/v
+:)2Adam/ConE3/kernel/v
:2Adam/ConE3/bias/v
+:)2Adam/ConA4/kernel/v
:2Adam/ConA4/bias/v
+:)2Adam/ConC4/kernel/v
:2Adam/ConC4/bias/v
+:)2Adam/ConE4/kernel/v
:2Adam/ConE4/bias/v
%:#???2Adam/HidA/kernel/v
:?2Adam/HidA/bias/v
&:$???2Adam/HidA2/kernel/v
:?2Adam/HidA2/bias/v
%:#
?#?2Adam/HidA3/kernel/v
:?2Adam/HidA3/bias/v
%:#
??2Adam/HidA4/kernel/v
:?2Adam/HidA4/bias/v
%:#??	?2Adam/HidC/kernel/v
:?2Adam/HidC/bias/v
&:$???2Adam/HidC2/kernel/v
:?2Adam/HidC2/bias/v
%:#
? ?2Adam/HidC3/kernel/v
:?2Adam/HidC3/bias/v
%:#
??2Adam/HidC4/kernel/v
:?2Adam/HidC4/bias/v
%:#???2Adam/HidE/kernel/v
:?2Adam/HidE/bias/v
&:$???2Adam/HidE2/kernel/v
:?2Adam/HidE2/bias/v
%:#
??2Adam/HidE3/kernel/v
:?2Adam/HidE3/bias/v
%:#
??2Adam/HidE4/kernel/v
:?2Adam/HidE4/bias/v
$:"
??2Adam/Hid1/kernel/v
:?2Adam/Hid1/bias/v
$:"
??2Adam/Hid2/kernel/v
:?2Adam/Hid2/bias/v
%:#	?(2Adam/output/kernel/v
:(2Adam/output/bias/v
?B?
!__inference__wrapped_model_226331input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_model_layer_call_fn_227444
&__inference_model_layer_call_fn_229240
&__inference_model_layer_call_fn_229353
&__inference_model_layer_call_fn_228654?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_model_layer_call_and_return_conditional_losses_229596
A__inference_model_layer_call_and_return_conditional_losses_229923
A__inference_model_layer_call_and_return_conditional_losses_228830
A__inference_model_layer_call_and_return_conditional_losses_229006?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
%__inference_ConA_layer_call_fn_229932?
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
@__inference_ConA_layer_call_and_return_conditional_losses_229943?
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
%__inference_ConC_layer_call_fn_229952?
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
@__inference_ConC_layer_call_and_return_conditional_losses_229963?
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
%__inference_ConE_layer_call_fn_229972?
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
@__inference_ConE_layer_call_and_return_conditional_losses_229983?
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
&__inference_PoolA_layer_call_fn_229988
&__inference_PoolA_layer_call_fn_229993?
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
A__inference_PoolA_layer_call_and_return_conditional_losses_229998
A__inference_PoolA_layer_call_and_return_conditional_losses_230003?
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
&__inference_PoolC_layer_call_fn_230008
&__inference_PoolC_layer_call_fn_230013?
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
A__inference_PoolC_layer_call_and_return_conditional_losses_230018
A__inference_PoolC_layer_call_and_return_conditional_losses_230023?
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
&__inference_PoolE_layer_call_fn_230028
&__inference_PoolE_layer_call_fn_230033?
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
A__inference_PoolE_layer_call_and_return_conditional_losses_230038
A__inference_PoolE_layer_call_and_return_conditional_losses_230043?
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
(__inference_dropout_layer_call_fn_230048
(__inference_dropout_layer_call_fn_230053?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dropout_layer_call_and_return_conditional_losses_230058
C__inference_dropout_layer_call_and_return_conditional_losses_230070?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_1_layer_call_fn_230075
*__inference_dropout_1_layer_call_fn_230080?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_1_layer_call_and_return_conditional_losses_230085
E__inference_dropout_1_layer_call_and_return_conditional_losses_230097?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_2_layer_call_fn_230102
*__inference_dropout_2_layer_call_fn_230107?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_2_layer_call_and_return_conditional_losses_230112
E__inference_dropout_2_layer_call_and_return_conditional_losses_230124?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_ConA2_layer_call_fn_230133?
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
A__inference_ConA2_layer_call_and_return_conditional_losses_230144?
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
&__inference_ConC2_layer_call_fn_230153?
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
A__inference_ConC2_layer_call_and_return_conditional_losses_230164?
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
&__inference_ConE2_layer_call_fn_230173?
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
A__inference_ConE2_layer_call_and_return_conditional_losses_230184?
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
'__inference_PoolA2_layer_call_fn_230189
'__inference_PoolA2_layer_call_fn_230194?
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
B__inference_PoolA2_layer_call_and_return_conditional_losses_230199
B__inference_PoolA2_layer_call_and_return_conditional_losses_230204?
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
'__inference_PoolC2_layer_call_fn_230209
'__inference_PoolC2_layer_call_fn_230214?
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
B__inference_PoolC2_layer_call_and_return_conditional_losses_230219
B__inference_PoolC2_layer_call_and_return_conditional_losses_230224?
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
'__inference_PoolE2_layer_call_fn_230229
'__inference_PoolE2_layer_call_fn_230234?
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
B__inference_PoolE2_layer_call_and_return_conditional_losses_230239
B__inference_PoolE2_layer_call_and_return_conditional_losses_230244?
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
&__inference_ConA3_layer_call_fn_230253?
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
A__inference_ConA3_layer_call_and_return_conditional_losses_230264?
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
&__inference_ConC3_layer_call_fn_230273?
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
A__inference_ConC3_layer_call_and_return_conditional_losses_230284?
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
&__inference_ConE3_layer_call_fn_230293?
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
A__inference_ConE3_layer_call_and_return_conditional_losses_230304?
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
'__inference_PoolA3_layer_call_fn_230309
'__inference_PoolA3_layer_call_fn_230314?
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
B__inference_PoolA3_layer_call_and_return_conditional_losses_230319
B__inference_PoolA3_layer_call_and_return_conditional_losses_230324?
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
'__inference_PoolC3_layer_call_fn_230329
'__inference_PoolC3_layer_call_fn_230334?
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
B__inference_PoolC3_layer_call_and_return_conditional_losses_230339
B__inference_PoolC3_layer_call_and_return_conditional_losses_230344?
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
'__inference_PoolE3_layer_call_fn_230349
'__inference_PoolE3_layer_call_fn_230354?
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
B__inference_PoolE3_layer_call_and_return_conditional_losses_230359
B__inference_PoolE3_layer_call_and_return_conditional_losses_230364?
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
&__inference_ConA4_layer_call_fn_230373?
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
A__inference_ConA4_layer_call_and_return_conditional_losses_230384?
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
&__inference_ConC4_layer_call_fn_230393?
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
A__inference_ConC4_layer_call_and_return_conditional_losses_230404?
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
&__inference_ConE4_layer_call_fn_230413?
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
A__inference_ConE4_layer_call_and_return_conditional_losses_230424?
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
'__inference_PoolA4_layer_call_fn_230429
'__inference_PoolA4_layer_call_fn_230434?
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
B__inference_PoolA4_layer_call_and_return_conditional_losses_230439
B__inference_PoolA4_layer_call_and_return_conditional_losses_230444?
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
'__inference_PoolC4_layer_call_fn_230449
'__inference_PoolC4_layer_call_fn_230454?
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
B__inference_PoolC4_layer_call_and_return_conditional_losses_230459
B__inference_PoolC4_layer_call_and_return_conditional_losses_230464?
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
'__inference_PoolE4_layer_call_fn_230469
'__inference_PoolE4_layer_call_fn_230474?
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
B__inference_PoolE4_layer_call_and_return_conditional_losses_230479
B__inference_PoolE4_layer_call_and_return_conditional_losses_230484?
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
*__inference_dropout_3_layer_call_fn_230489
*__inference_dropout_3_layer_call_fn_230494?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_3_layer_call_and_return_conditional_losses_230499
E__inference_dropout_3_layer_call_and_return_conditional_losses_230511?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_4_layer_call_fn_230516
*__inference_dropout_4_layer_call_fn_230521?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_4_layer_call_and_return_conditional_losses_230526
E__inference_dropout_4_layer_call_and_return_conditional_losses_230538?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_5_layer_call_fn_230543
*__inference_dropout_5_layer_call_fn_230548?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_5_layer_call_and_return_conditional_losses_230553
E__inference_dropout_5_layer_call_and_return_conditional_losses_230565?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_6_layer_call_fn_230570
*__inference_dropout_6_layer_call_fn_230575?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_6_layer_call_and_return_conditional_losses_230580
E__inference_dropout_6_layer_call_and_return_conditional_losses_230592?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_7_layer_call_fn_230597
*__inference_dropout_7_layer_call_fn_230602?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_7_layer_call_and_return_conditional_losses_230607
E__inference_dropout_7_layer_call_and_return_conditional_losses_230619?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_8_layer_call_fn_230624
*__inference_dropout_8_layer_call_fn_230629?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_8_layer_call_and_return_conditional_losses_230634
E__inference_dropout_8_layer_call_and_return_conditional_losses_230646?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_9_layer_call_fn_230651
*__inference_dropout_9_layer_call_fn_230656?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_9_layer_call_and_return_conditional_losses_230661
E__inference_dropout_9_layer_call_and_return_conditional_losses_230673?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout_10_layer_call_fn_230678
+__inference_dropout_10_layer_call_fn_230683?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_10_layer_call_and_return_conditional_losses_230688
F__inference_dropout_10_layer_call_and_return_conditional_losses_230700?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout_11_layer_call_fn_230705
+__inference_dropout_11_layer_call_fn_230710?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_11_layer_call_and_return_conditional_losses_230715
F__inference_dropout_11_layer_call_and_return_conditional_losses_230727?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_flatten_layer_call_fn_230732?
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
C__inference_flatten_layer_call_and_return_conditional_losses_230738?
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
*__inference_flatten_1_layer_call_fn_230743?
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
E__inference_flatten_1_layer_call_and_return_conditional_losses_230749?
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
*__inference_flatten_2_layer_call_fn_230754?
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
E__inference_flatten_2_layer_call_and_return_conditional_losses_230760?
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
*__inference_flatten_3_layer_call_fn_230765?
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
E__inference_flatten_3_layer_call_and_return_conditional_losses_230771?
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
*__inference_flatten_4_layer_call_fn_230776?
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
E__inference_flatten_4_layer_call_and_return_conditional_losses_230782?
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
*__inference_flatten_5_layer_call_fn_230787?
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
E__inference_flatten_5_layer_call_and_return_conditional_losses_230793?
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
*__inference_flatten_6_layer_call_fn_230798?
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
E__inference_flatten_6_layer_call_and_return_conditional_losses_230804?
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
*__inference_flatten_7_layer_call_fn_230809?
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
E__inference_flatten_7_layer_call_and_return_conditional_losses_230815?
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
*__inference_flatten_8_layer_call_fn_230820?
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
E__inference_flatten_8_layer_call_and_return_conditional_losses_230826?
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
*__inference_flatten_9_layer_call_fn_230831?
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
E__inference_flatten_9_layer_call_and_return_conditional_losses_230837?
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
+__inference_flatten_10_layer_call_fn_230842?
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
F__inference_flatten_10_layer_call_and_return_conditional_losses_230848?
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
+__inference_flatten_11_layer_call_fn_230853?
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
F__inference_flatten_11_layer_call_and_return_conditional_losses_230859?
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
%__inference_HidA_layer_call_fn_230868?
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
@__inference_HidA_layer_call_and_return_conditional_losses_230879?
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
&__inference_HidA2_layer_call_fn_230888?
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
A__inference_HidA2_layer_call_and_return_conditional_losses_230899?
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
&__inference_HidA3_layer_call_fn_230908?
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
A__inference_HidA3_layer_call_and_return_conditional_losses_230919?
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
&__inference_HidA4_layer_call_fn_230928?
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
A__inference_HidA4_layer_call_and_return_conditional_losses_230939?
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
%__inference_HidC_layer_call_fn_230948?
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
@__inference_HidC_layer_call_and_return_conditional_losses_230959?
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
&__inference_HidC2_layer_call_fn_230968?
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
A__inference_HidC2_layer_call_and_return_conditional_losses_230979?
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
&__inference_HidC3_layer_call_fn_230988?
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
A__inference_HidC3_layer_call_and_return_conditional_losses_230999?
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
&__inference_HidC4_layer_call_fn_231008?
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
A__inference_HidC4_layer_call_and_return_conditional_losses_231019?
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
%__inference_HidE_layer_call_fn_231028?
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
@__inference_HidE_layer_call_and_return_conditional_losses_231039?
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
&__inference_HidE2_layer_call_fn_231048?
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
A__inference_HidE2_layer_call_and_return_conditional_losses_231059?
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
&__inference_HidE3_layer_call_fn_231068?
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
A__inference_HidE3_layer_call_and_return_conditional_losses_231079?
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
&__inference_HidE4_layer_call_fn_231088?
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
A__inference_HidE4_layer_call_and_return_conditional_losses_231099?
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
,__inference_concatenate_layer_call_fn_231115?
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
G__inference_concatenate_layer_call_and_return_conditional_losses_231132?
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
%__inference_Hid1_layer_call_fn_231141?
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
@__inference_Hid1_layer_call_and_return_conditional_losses_231152?
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
%__inference_Hid2_layer_call_fn_231161?
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
@__inference_Hid2_layer_call_and_return_conditional_losses_231172?
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
'__inference_output_layer_call_fn_231181?
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
B__inference_output_layer_call_and_return_conditional_losses_231192?
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
?B?
$__inference_signature_wrapper_229127input"?
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
 ?
A__inference_ConA2_layer_call_and_return_conditional_losses_230144l}~7?4
-?*
(?%
inputs?????????I?
? "-?*
#? 
0?????????G=
? ?
&__inference_ConA2_layer_call_fn_230133_}~7?4
-?*
(?%
inputs?????????I?
? " ??????????G=?
A__inference_ConA3_layer_call_and_return_conditional_losses_230264n??7?4
-?*
(?%
inputs?????????#
? "-?*
#? 
0?????????!
? ?
&__inference_ConA3_layer_call_fn_230253a??7?4
-?*
(?%
inputs?????????#
? " ??????????!?
A__inference_ConA4_layer_call_and_return_conditional_losses_230384n??7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
&__inference_ConA4_layer_call_fn_230373a??7?4
-?*
(?%
inputs?????????
? " ???????????
@__inference_ConA_layer_call_and_return_conditional_losses_229943pJK9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
%__inference_ConA_layer_call_fn_229932cJK9?6
/?,
*?'
inputs???????????
? ""?????????????
A__inference_ConC2_layer_call_and_return_conditional_losses_230164n??7?4
-?*
(?%
inputs?????????F=$
? "-?*
#? 
0?????????D;
? ?
&__inference_ConC2_layer_call_fn_230153a??7?4
-?*
(?%
inputs?????????F=$
? " ??????????D;?
A__inference_ConC3_layer_call_and_return_conditional_losses_230284n??7?4
-?*
(?%
inputs?????????"
? "-?*
#? 
0????????? 
? ?
&__inference_ConC3_layer_call_fn_230273a??7?4
-?*
(?%
inputs?????????"
? " ?????????? ?
A__inference_ConC4_layer_call_and_return_conditional_losses_230404n??7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
&__inference_ConC4_layer_call_fn_230393a??7?4
-?*
(?%
inputs?????????
? " ???????????
@__inference_ConC_layer_call_and_return_conditional_losses_229963pQR9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????$
? ?
%__inference_ConC_layer_call_fn_229952cQR9?6
/?,
*?'
inputs???????????
? ""????????????$?
A__inference_ConE2_layer_call_and_return_conditional_losses_230184n??7?4
-?*
(?%
inputs?????????B:0
? "-?*
#? 
0?????????@8
? ?
&__inference_ConE2_layer_call_fn_230173a??7?4
-?*
(?%
inputs?????????B:0
? " ??????????@8?
A__inference_ConE3_layer_call_and_return_conditional_losses_230304n??7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????
? ?
&__inference_ConE3_layer_call_fn_230293a??7?4
-?*
(?%
inputs????????? 
? " ???????????
A__inference_ConE4_layer_call_and_return_conditional_losses_230424n??7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
&__inference_ConE4_layer_call_fn_230413a??7?4
-?*
(?%
inputs?????????
? " ???????????
@__inference_ConE_layer_call_and_return_conditional_losses_229983pXY9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????0
? ?
%__inference_ConE_layer_call_fn_229972cXY9?6
/?,
*?'
inputs???????????
? ""????????????0?
@__inference_Hid1_layer_call_and_return_conditional_losses_231152`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? |
%__inference_Hid1_layer_call_fn_231141S??0?-
&?#
!?
inputs??????????
? "????????????
@__inference_Hid2_layer_call_and_return_conditional_losses_231172`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? |
%__inference_Hid2_layer_call_fn_231161S??0?-
&?#
!?
inputs??????????
? "????????????
A__inference_HidA2_layer_call_and_return_conditional_losses_230899a??1?.
'?$
"?
inputs???????????
? "&?#
?
0??????????
? ~
&__inference_HidA2_layer_call_fn_230888T??1?.
'?$
"?
inputs???????????
? "????????????
A__inference_HidA3_layer_call_and_return_conditional_losses_230919`??0?-
&?#
!?
inputs??????????#
? "&?#
?
0??????????
? }
&__inference_HidA3_layer_call_fn_230908S??0?-
&?#
!?
inputs??????????#
? "????????????
A__inference_HidA4_layer_call_and_return_conditional_losses_230939`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
&__inference_HidA4_layer_call_fn_230928S??0?-
&?#
!?
inputs??????????
? "????????????
@__inference_HidA_layer_call_and_return_conditional_losses_230879a??1?.
'?$
"?
inputs???????????
? "&?#
?
0??????????
? }
%__inference_HidA_layer_call_fn_230868T??1?.
'?$
"?
inputs???????????
? "????????????
A__inference_HidC2_layer_call_and_return_conditional_losses_230979a??1?.
'?$
"?
inputs???????????
? "&?#
?
0??????????
? ~
&__inference_HidC2_layer_call_fn_230968T??1?.
'?$
"?
inputs???????????
? "????????????
A__inference_HidC3_layer_call_and_return_conditional_losses_230999`??0?-
&?#
!?
inputs?????????? 
? "&?#
?
0??????????
? }
&__inference_HidC3_layer_call_fn_230988S??0?-
&?#
!?
inputs?????????? 
? "????????????
A__inference_HidC4_layer_call_and_return_conditional_losses_231019`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
&__inference_HidC4_layer_call_fn_231008S??0?-
&?#
!?
inputs??????????
? "????????????
@__inference_HidC_layer_call_and_return_conditional_losses_230959a??1?.
'?$
"?
inputs???????????	
? "&?#
?
0??????????
? }
%__inference_HidC_layer_call_fn_230948T??1?.
'?$
"?
inputs???????????	
? "????????????
A__inference_HidE2_layer_call_and_return_conditional_losses_231059a??1?.
'?$
"?
inputs???????????
? "&?#
?
0??????????
? ~
&__inference_HidE2_layer_call_fn_231048T??1?.
'?$
"?
inputs???????????
? "????????????
A__inference_HidE3_layer_call_and_return_conditional_losses_231079`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
&__inference_HidE3_layer_call_fn_231068S??0?-
&?#
!?
inputs??????????
? "????????????
A__inference_HidE4_layer_call_and_return_conditional_losses_231099`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
&__inference_HidE4_layer_call_fn_231088S??0?-
&?#
!?
inputs??????????
? "????????????
@__inference_HidE_layer_call_and_return_conditional_losses_231039a??1?.
'?$
"?
inputs???????????
? "&?#
?
0??????????
? }
%__inference_HidE_layer_call_fn_231028T??1?.
'?$
"?
inputs???????????
? "????????????
B__inference_PoolA2_layer_call_and_return_conditional_losses_230199?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
B__inference_PoolA2_layer_call_and_return_conditional_losses_230204h7?4
-?*
(?%
inputs?????????G=
? "-?*
#? 
0?????????#
? ?
'__inference_PoolA2_layer_call_fn_230189?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
'__inference_PoolA2_layer_call_fn_230194[7?4
-?*
(?%
inputs?????????G=
? " ??????????#?
B__inference_PoolA3_layer_call_and_return_conditional_losses_230319?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
B__inference_PoolA3_layer_call_and_return_conditional_losses_230324h7?4
-?*
(?%
inputs?????????!
? "-?*
#? 
0?????????
? ?
'__inference_PoolA3_layer_call_fn_230309?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
'__inference_PoolA3_layer_call_fn_230314[7?4
-?*
(?%
inputs?????????!
? " ???????????
B__inference_PoolA4_layer_call_and_return_conditional_losses_230439?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
B__inference_PoolA4_layer_call_and_return_conditional_losses_230444h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
'__inference_PoolA4_layer_call_fn_230429?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
'__inference_PoolA4_layer_call_fn_230434[7?4
-?*
(?%
inputs?????????
? " ???????????
A__inference_PoolA_layer_call_and_return_conditional_losses_229998?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
A__inference_PoolA_layer_call_and_return_conditional_losses_230003j9?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????I?
? ?
&__inference_PoolA_layer_call_fn_229988?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
&__inference_PoolA_layer_call_fn_229993]9?6
/?,
*?'
inputs???????????
? " ??????????I??
B__inference_PoolC2_layer_call_and_return_conditional_losses_230219?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
B__inference_PoolC2_layer_call_and_return_conditional_losses_230224h7?4
-?*
(?%
inputs?????????D;
? "-?*
#? 
0?????????"
? ?
'__inference_PoolC2_layer_call_fn_230209?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
'__inference_PoolC2_layer_call_fn_230214[7?4
-?*
(?%
inputs?????????D;
? " ??????????"?
B__inference_PoolC3_layer_call_and_return_conditional_losses_230339?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
B__inference_PoolC3_layer_call_and_return_conditional_losses_230344h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????
? ?
'__inference_PoolC3_layer_call_fn_230329?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
'__inference_PoolC3_layer_call_fn_230334[7?4
-?*
(?%
inputs????????? 
? " ???????????
B__inference_PoolC4_layer_call_and_return_conditional_losses_230459?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
B__inference_PoolC4_layer_call_and_return_conditional_losses_230464h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
'__inference_PoolC4_layer_call_fn_230449?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
'__inference_PoolC4_layer_call_fn_230454[7?4
-?*
(?%
inputs?????????
? " ???????????
A__inference_PoolC_layer_call_and_return_conditional_losses_230018?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
A__inference_PoolC_layer_call_and_return_conditional_losses_230023j9?6
/?,
*?'
inputs???????????$
? "-?*
#? 
0?????????F=$
? ?
&__inference_PoolC_layer_call_fn_230008?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
&__inference_PoolC_layer_call_fn_230013]9?6
/?,
*?'
inputs???????????$
? " ??????????F=$?
B__inference_PoolE2_layer_call_and_return_conditional_losses_230239?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
B__inference_PoolE2_layer_call_and_return_conditional_losses_230244h7?4
-?*
(?%
inputs?????????@8
? "-?*
#? 
0????????? 
? ?
'__inference_PoolE2_layer_call_fn_230229?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
'__inference_PoolE2_layer_call_fn_230234[7?4
-?*
(?%
inputs?????????@8
? " ?????????? ?
B__inference_PoolE3_layer_call_and_return_conditional_losses_230359?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
B__inference_PoolE3_layer_call_and_return_conditional_losses_230364h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
'__inference_PoolE3_layer_call_fn_230349?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
'__inference_PoolE3_layer_call_fn_230354[7?4
-?*
(?%
inputs?????????
? " ???????????
B__inference_PoolE4_layer_call_and_return_conditional_losses_230479?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
B__inference_PoolE4_layer_call_and_return_conditional_losses_230484h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
'__inference_PoolE4_layer_call_fn_230469?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
'__inference_PoolE4_layer_call_fn_230474[7?4
-?*
(?%
inputs?????????
? " ???????????
A__inference_PoolE_layer_call_and_return_conditional_losses_230038?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
A__inference_PoolE_layer_call_and_return_conditional_losses_230043j9?6
/?,
*?'
inputs???????????0
? "-?*
#? 
0?????????B:0
? ?
&__inference_PoolE_layer_call_fn_230028?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
&__inference_PoolE_layer_call_fn_230033]9?6
/?,
*?'
inputs???????????0
? " ??????????B:0?
!__inference__wrapped_model_226331?dXYQRJK????}~??????????????????????????????????????????8?5
.?+
)?&
input???????????
? "/?,
*
output ?
output?????????(?
G__inference_concatenate_layer_call_and_return_conditional_losses_231132????
???
???
#? 
inputs/0??????????
#? 
inputs/1??????????
#? 
inputs/2??????????
#? 
inputs/3??????????
#? 
inputs/4??????????
#? 
inputs/5??????????
#? 
inputs/6??????????
#? 
inputs/7??????????
#? 
inputs/8??????????
#? 
inputs/9??????????
$?!
	inputs/10??????????
$?!
	inputs/11??????????
? "&?#
?
0??????????
? ?
,__inference_concatenate_layer_call_fn_231115????
???
???
#? 
inputs/0??????????
#? 
inputs/1??????????
#? 
inputs/2??????????
#? 
inputs/3??????????
#? 
inputs/4??????????
#? 
inputs/5??????????
#? 
inputs/6??????????
#? 
inputs/7??????????
#? 
inputs/8??????????
#? 
inputs/9??????????
$?!
	inputs/10??????????
$?!
	inputs/11??????????
? "????????????
F__inference_dropout_10_layer_call_and_return_conditional_losses_230688l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
F__inference_dropout_10_layer_call_and_return_conditional_losses_230700l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
+__inference_dropout_10_layer_call_fn_230678_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
+__inference_dropout_10_layer_call_fn_230683_;?8
1?.
(?%
inputs?????????
p
? " ???????????
F__inference_dropout_11_layer_call_and_return_conditional_losses_230715l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
F__inference_dropout_11_layer_call_and_return_conditional_losses_230727l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
+__inference_dropout_11_layer_call_fn_230705_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
+__inference_dropout_11_layer_call_fn_230710_;?8
1?.
(?%
inputs?????????
p
? " ???????????
E__inference_dropout_1_layer_call_and_return_conditional_losses_230085l;?8
1?.
(?%
inputs?????????F=$
p 
? "-?*
#? 
0?????????F=$
? ?
E__inference_dropout_1_layer_call_and_return_conditional_losses_230097l;?8
1?.
(?%
inputs?????????F=$
p
? "-?*
#? 
0?????????F=$
? ?
*__inference_dropout_1_layer_call_fn_230075_;?8
1?.
(?%
inputs?????????F=$
p 
? " ??????????F=$?
*__inference_dropout_1_layer_call_fn_230080_;?8
1?.
(?%
inputs?????????F=$
p
? " ??????????F=$?
E__inference_dropout_2_layer_call_and_return_conditional_losses_230112l;?8
1?.
(?%
inputs?????????B:0
p 
? "-?*
#? 
0?????????B:0
? ?
E__inference_dropout_2_layer_call_and_return_conditional_losses_230124l;?8
1?.
(?%
inputs?????????B:0
p
? "-?*
#? 
0?????????B:0
? ?
*__inference_dropout_2_layer_call_fn_230102_;?8
1?.
(?%
inputs?????????B:0
p 
? " ??????????B:0?
*__inference_dropout_2_layer_call_fn_230107_;?8
1?.
(?%
inputs?????????B:0
p
? " ??????????B:0?
E__inference_dropout_3_layer_call_and_return_conditional_losses_230499l;?8
1?.
(?%
inputs?????????#
p 
? "-?*
#? 
0?????????#
? ?
E__inference_dropout_3_layer_call_and_return_conditional_losses_230511l;?8
1?.
(?%
inputs?????????#
p
? "-?*
#? 
0?????????#
? ?
*__inference_dropout_3_layer_call_fn_230489_;?8
1?.
(?%
inputs?????????#
p 
? " ??????????#?
*__inference_dropout_3_layer_call_fn_230494_;?8
1?.
(?%
inputs?????????#
p
? " ??????????#?
E__inference_dropout_4_layer_call_and_return_conditional_losses_230526l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
E__inference_dropout_4_layer_call_and_return_conditional_losses_230538l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
*__inference_dropout_4_layer_call_fn_230516_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
*__inference_dropout_4_layer_call_fn_230521_;?8
1?.
(?%
inputs?????????
p
? " ???????????
E__inference_dropout_5_layer_call_and_return_conditional_losses_230553l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
E__inference_dropout_5_layer_call_and_return_conditional_losses_230565l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
*__inference_dropout_5_layer_call_fn_230543_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
*__inference_dropout_5_layer_call_fn_230548_;?8
1?.
(?%
inputs?????????
p
? " ???????????
E__inference_dropout_6_layer_call_and_return_conditional_losses_230580l;?8
1?.
(?%
inputs?????????"
p 
? "-?*
#? 
0?????????"
? ?
E__inference_dropout_6_layer_call_and_return_conditional_losses_230592l;?8
1?.
(?%
inputs?????????"
p
? "-?*
#? 
0?????????"
? ?
*__inference_dropout_6_layer_call_fn_230570_;?8
1?.
(?%
inputs?????????"
p 
? " ??????????"?
*__inference_dropout_6_layer_call_fn_230575_;?8
1?.
(?%
inputs?????????"
p
? " ??????????"?
E__inference_dropout_7_layer_call_and_return_conditional_losses_230607l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
E__inference_dropout_7_layer_call_and_return_conditional_losses_230619l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
*__inference_dropout_7_layer_call_fn_230597_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
*__inference_dropout_7_layer_call_fn_230602_;?8
1?.
(?%
inputs?????????
p
? " ???????????
E__inference_dropout_8_layer_call_and_return_conditional_losses_230634l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
E__inference_dropout_8_layer_call_and_return_conditional_losses_230646l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
*__inference_dropout_8_layer_call_fn_230624_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
*__inference_dropout_8_layer_call_fn_230629_;?8
1?.
(?%
inputs?????????
p
? " ???????????
E__inference_dropout_9_layer_call_and_return_conditional_losses_230661l;?8
1?.
(?%
inputs????????? 
p 
? "-?*
#? 
0????????? 
? ?
E__inference_dropout_9_layer_call_and_return_conditional_losses_230673l;?8
1?.
(?%
inputs????????? 
p
? "-?*
#? 
0????????? 
? ?
*__inference_dropout_9_layer_call_fn_230651_;?8
1?.
(?%
inputs????????? 
p 
? " ?????????? ?
*__inference_dropout_9_layer_call_fn_230656_;?8
1?.
(?%
inputs????????? 
p
? " ?????????? ?
C__inference_dropout_layer_call_and_return_conditional_losses_230058l;?8
1?.
(?%
inputs?????????I?
p 
? "-?*
#? 
0?????????I?
? ?
C__inference_dropout_layer_call_and_return_conditional_losses_230070l;?8
1?.
(?%
inputs?????????I?
p
? "-?*
#? 
0?????????I?
? ?
(__inference_dropout_layer_call_fn_230048_;?8
1?.
(?%
inputs?????????I?
p 
? " ??????????I??
(__inference_dropout_layer_call_fn_230053_;?8
1?.
(?%
inputs?????????I?
p
? " ??????????I??
F__inference_flatten_10_layer_call_and_return_conditional_losses_230848a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
+__inference_flatten_10_layer_call_fn_230842T7?4
-?*
(?%
inputs?????????
? "????????????
F__inference_flatten_11_layer_call_and_return_conditional_losses_230859a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
+__inference_flatten_11_layer_call_fn_230853T7?4
-?*
(?%
inputs?????????
? "????????????
E__inference_flatten_1_layer_call_and_return_conditional_losses_230749b7?4
-?*
(?%
inputs?????????#
? "'?$
?
0???????????
? ?
*__inference_flatten_1_layer_call_fn_230743U7?4
-?*
(?%
inputs?????????#
? "?????????????
E__inference_flatten_2_layer_call_and_return_conditional_losses_230760a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????#
? ?
*__inference_flatten_2_layer_call_fn_230754T7?4
-?*
(?%
inputs?????????
? "???????????#?
E__inference_flatten_3_layer_call_and_return_conditional_losses_230771a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
*__inference_flatten_3_layer_call_fn_230765T7?4
-?*
(?%
inputs?????????
? "????????????
E__inference_flatten_4_layer_call_and_return_conditional_losses_230782b7?4
-?*
(?%
inputs?????????F=$
? "'?$
?
0???????????	
? ?
*__inference_flatten_4_layer_call_fn_230776U7?4
-?*
(?%
inputs?????????F=$
? "????????????	?
E__inference_flatten_5_layer_call_and_return_conditional_losses_230793b7?4
-?*
(?%
inputs?????????"
? "'?$
?
0???????????
? ?
*__inference_flatten_5_layer_call_fn_230787U7?4
-?*
(?%
inputs?????????"
? "?????????????
E__inference_flatten_6_layer_call_and_return_conditional_losses_230804a7?4
-?*
(?%
inputs?????????
? "&?#
?
0?????????? 
? ?
*__inference_flatten_6_layer_call_fn_230798T7?4
-?*
(?%
inputs?????????
? "??????????? ?
E__inference_flatten_7_layer_call_and_return_conditional_losses_230815a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
*__inference_flatten_7_layer_call_fn_230809T7?4
-?*
(?%
inputs?????????
? "????????????
E__inference_flatten_8_layer_call_and_return_conditional_losses_230826b7?4
-?*
(?%
inputs?????????B:0
? "'?$
?
0???????????
? ?
*__inference_flatten_8_layer_call_fn_230820U7?4
-?*
(?%
inputs?????????B:0
? "?????????????
E__inference_flatten_9_layer_call_and_return_conditional_losses_230837b7?4
-?*
(?%
inputs????????? 
? "'?$
?
0???????????
? ?
*__inference_flatten_9_layer_call_fn_230831U7?4
-?*
(?%
inputs????????? 
? "?????????????
C__inference_flatten_layer_call_and_return_conditional_losses_230738b7?4
-?*
(?%
inputs?????????I?
? "'?$
?
0???????????
? ?
(__inference_flatten_layer_call_fn_230732U7?4
-?*
(?%
inputs?????????I?
? "?????????????
A__inference_model_layer_call_and_return_conditional_losses_228830?dXYQRJK????}~??????????????????????????????????????????@?=
6?3
)?&
input???????????
p 

 
? "%?"
?
0?????????(
? ?
A__inference_model_layer_call_and_return_conditional_losses_229006?dXYQRJK????}~??????????????????????????????????????????@?=
6?3
)?&
input???????????
p

 
? "%?"
?
0?????????(
? ?
A__inference_model_layer_call_and_return_conditional_losses_229596?dXYQRJK????}~??????????????????????????????????????????A?>
7?4
*?'
inputs???????????
p 

 
? "%?"
?
0?????????(
? ?
A__inference_model_layer_call_and_return_conditional_losses_229923?dXYQRJK????}~??????????????????????????????????????????A?>
7?4
*?'
inputs???????????
p

 
? "%?"
?
0?????????(
? ?
&__inference_model_layer_call_fn_227444?dXYQRJK????}~??????????????????????????????????????????@?=
6?3
)?&
input???????????
p 

 
? "??????????(?
&__inference_model_layer_call_fn_228654?dXYQRJK????}~??????????????????????????????????????????@?=
6?3
)?&
input???????????
p

 
? "??????????(?
&__inference_model_layer_call_fn_229240?dXYQRJK????}~??????????????????????????????????????????A?>
7?4
*?'
inputs???????????
p 

 
? "??????????(?
&__inference_model_layer_call_fn_229353?dXYQRJK????}~??????????????????????????????????????????A?>
7?4
*?'
inputs???????????
p

 
? "??????????(?
B__inference_output_layer_call_and_return_conditional_losses_231192_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????(
? }
'__inference_output_layer_call_fn_231181R??0?-
&?#
!?
inputs??????????
? "??????????(?
$__inference_signature_wrapper_229127?dXYQRJK????}~??????????????????????????????????????????A?>
? 
7?4
2
input)?&
input???????????"/?,
*
output ?
output?????????(