??2
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
 ?"serve*2.6.22v2.6.1-9-gc2363d6d0258??+
z
ConA/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConA/kernel
s
ConA/kernel/Read/ReadVariableOpReadVariableOpConA/kernel*&
_output_shapes
:*
dtype0
j
	ConA/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	ConA/bias
c
ConA/bias/Read/ReadVariableOpReadVariableOp	ConA/bias*
_output_shapes
:*
dtype0
z
ConB/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConB/kernel
s
ConB/kernel/Read/ReadVariableOpReadVariableOpConB/kernel*&
_output_shapes
:*
dtype0
j
	ConB/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	ConB/bias
c
ConB/bias/Read/ReadVariableOpReadVariableOp	ConB/bias*
_output_shapes
:*
dtype0
|
ConA2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConA2/kernel
u
 ConA2/kernel/Read/ReadVariableOpReadVariableOpConA2/kernel*&
_output_shapes
:*
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
z
ConC/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameConC/kernel
s
ConC/kernel/Read/ReadVariableOpReadVariableOpConC/kernel*&
_output_shapes
:	*
dtype0
j
	ConC/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	ConC/bias
c
ConC/bias/Read/ReadVariableOpReadVariableOp	ConC/bias*
_output_shapes
:*
dtype0
|
ConA3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameConA3/kernel
u
 ConA3/kernel/Read/ReadVariableOpReadVariableOpConA3/kernel*&
_output_shapes
:(*
dtype0
l

ConA3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_name
ConA3/bias
e
ConA3/bias/Read/ReadVariableOpReadVariableOp
ConA3/bias*
_output_shapes
:(*
dtype0
|
ConB2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConB2/kernel
u
 ConB2/kernel/Read/ReadVariableOpReadVariableOpConB2/kernel*&
_output_shapes
:*
dtype0
l

ConB2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
ConB2/bias
e
ConB2/bias/Read/ReadVariableOpReadVariableOp
ConB2/bias*
_output_shapes
:*
dtype0
|
ConA4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:((*
shared_nameConA4/kernel
u
 ConA4/kernel/Read/ReadVariableOpReadVariableOpConA4/kernel*&
_output_shapes
:((*
dtype0
l

ConA4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_name
ConA4/bias
e
ConA4/bias/Read/ReadVariableOpReadVariableOp
ConA4/bias*
_output_shapes
:(*
dtype0
|
ConB3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameConB3/kernel
u
 ConB3/kernel/Read/ReadVariableOpReadVariableOpConB3/kernel*&
_output_shapes
:(*
dtype0
l

ConB3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_name
ConB3/bias
e
ConB3/bias/Read/ReadVariableOpReadVariableOp
ConB3/bias*
_output_shapes
:(*
dtype0
|
ConC2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConC2/kernel
u
 ConC2/kernel/Read/ReadVariableOpReadVariableOpConC2/kernel*&
_output_shapes
:*
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
z
ConD/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameConD/kernel
s
ConD/kernel/Read/ReadVariableOpReadVariableOpConD/kernel*&
_output_shapes
:	*
dtype0
j
	ConD/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	ConD/bias
c
ConD/bias/Read/ReadVariableOpReadVariableOp	ConD/bias*
_output_shapes
:*
dtype0
z
ConE/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameConE/kernel
s
ConE/kernel/Read/ReadVariableOpReadVariableOpConE/kernel*&
_output_shapes
: *
dtype0
j
	ConE/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	ConE/bias
c
ConE/bias/Read/ReadVariableOpReadVariableOp	ConE/bias*
_output_shapes
: *
dtype0
u
HidA/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*
shared_nameHidA/kernel
n
HidA/kernel/Read/ReadVariableOpReadVariableOpHidA/kernel*!
_output_shapes
:???*
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
?F?*
shared_nameHidA3/kernel
o
 HidA3/kernel/Read/ReadVariableOpReadVariableOpHidA3/kernel* 
_output_shapes
:
?F?*
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
??*
shared_nameHidA4/kernel
o
 HidA4/kernel/Read/ReadVariableOpReadVariableOpHidA4/kernel* 
_output_shapes
:
??*
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
HidB/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ȓ?*
shared_nameHidB/kernel
n
HidB/kernel/Read/ReadVariableOpReadVariableOpHidB/kernel*!
_output_shapes
:ȓ?*
dtype0
k
	HidB/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	HidB/bias
d
HidB/bias/Read/ReadVariableOpReadVariableOp	HidB/bias*
_output_shapes	
:?*
dtype0
w
HidB2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*
shared_nameHidB2/kernel
p
 HidB2/kernel/Read/ReadVariableOpReadVariableOpHidB2/kernel*!
_output_shapes
:???*
dtype0
m

HidB2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
HidB2/bias
f
HidB2/bias/Read/ReadVariableOpReadVariableOp
HidB2/bias*
_output_shapes	
:?*
dtype0
v
HidB3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?F?*
shared_nameHidB3/kernel
o
 HidB3/kernel/Read/ReadVariableOpReadVariableOpHidB3/kernel* 
_output_shapes
:
?F?*
dtype0
m

HidB3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
HidB3/bias
f
HidB3/bias/Read/ReadVariableOpReadVariableOp
HidB3/bias*
_output_shapes	
:?*
dtype0
u
HidC/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*
shared_nameHidC/kernel
n
HidC/kernel/Read/ReadVariableOpReadVariableOpHidC/kernel*!
_output_shapes
:???*
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
w
HidD1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*
shared_nameHidD1/kernel
p
 HidD1/kernel/Read/ReadVariableOpReadVariableOpHidD1/kernel*!
_output_shapes
:???*
dtype0
m

HidD1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
HidD1/bias
f
HidD1/bias/Read/ReadVariableOpReadVariableOp
HidD1/bias*
_output_shapes	
:?*
dtype0
w
HidE1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*
shared_nameHidE1/kernel
p
 HidE1/kernel/Read/ReadVariableOpReadVariableOpHidE1/kernel*!
_output_shapes
:???*
dtype0
m

HidE1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
HidE1/bias
f
HidE1/bias/Read/ReadVariableOpReadVariableOp
HidE1/bias*
_output_shapes	
:?*
dtype0
t
Hid1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameHid1/kernel
m
Hid1/kernel/Read/ReadVariableOpReadVariableOpHid1/kernel* 
_output_shapes
:
??*
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
x
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameoutput/kernel
q
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel* 
_output_shapes
:
??*
dtype0
o
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameoutput/bias
h
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes	
:?*
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
shape:*#
shared_nameAdam/ConA/kernel/m
?
&Adam/ConA/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConA/kernel/m*&
_output_shapes
:*
dtype0
x
Adam/ConA/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/ConA/bias/m
q
$Adam/ConA/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConA/bias/m*
_output_shapes
:*
dtype0
?
Adam/ConB/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/ConB/kernel/m
?
&Adam/ConB/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConB/kernel/m*&
_output_shapes
:*
dtype0
x
Adam/ConB/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/ConB/bias/m
q
$Adam/ConB/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConB/bias/m*
_output_shapes
:*
dtype0
?
Adam/ConA2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConA2/kernel/m
?
'Adam/ConA2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConA2/kernel/m*&
_output_shapes
:*
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
Adam/ConC/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*#
shared_nameAdam/ConC/kernel/m
?
&Adam/ConC/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConC/kernel/m*&
_output_shapes
:	*
dtype0
x
Adam/ConC/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/ConC/bias/m
q
$Adam/ConC/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConC/bias/m*
_output_shapes
:*
dtype0
?
Adam/ConA3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*$
shared_nameAdam/ConA3/kernel/m
?
'Adam/ConA3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConA3/kernel/m*&
_output_shapes
:(*
dtype0
z
Adam/ConA3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*"
shared_nameAdam/ConA3/bias/m
s
%Adam/ConA3/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConA3/bias/m*
_output_shapes
:(*
dtype0
?
Adam/ConB2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConB2/kernel/m
?
'Adam/ConB2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConB2/kernel/m*&
_output_shapes
:*
dtype0
z
Adam/ConB2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/ConB2/bias/m
s
%Adam/ConB2/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConB2/bias/m*
_output_shapes
:*
dtype0
?
Adam/ConA4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:((*$
shared_nameAdam/ConA4/kernel/m
?
'Adam/ConA4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConA4/kernel/m*&
_output_shapes
:((*
dtype0
z
Adam/ConA4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*"
shared_nameAdam/ConA4/bias/m
s
%Adam/ConA4/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConA4/bias/m*
_output_shapes
:(*
dtype0
?
Adam/ConB3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*$
shared_nameAdam/ConB3/kernel/m
?
'Adam/ConB3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConB3/kernel/m*&
_output_shapes
:(*
dtype0
z
Adam/ConB3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*"
shared_nameAdam/ConB3/bias/m
s
%Adam/ConB3/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConB3/bias/m*
_output_shapes
:(*
dtype0
?
Adam/ConC2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConC2/kernel/m
?
'Adam/ConC2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConC2/kernel/m*&
_output_shapes
:*
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
Adam/ConD/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*#
shared_nameAdam/ConD/kernel/m
?
&Adam/ConD/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConD/kernel/m*&
_output_shapes
:	*
dtype0
x
Adam/ConD/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/ConD/bias/m
q
$Adam/ConD/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConD/bias/m*
_output_shapes
:*
dtype0
?
Adam/ConE/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/ConE/kernel/m
?
&Adam/ConE/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConE/kernel/m*&
_output_shapes
: *
dtype0
x
Adam/ConE/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/ConE/bias/m
q
$Adam/ConE/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConE/bias/m*
_output_shapes
: *
dtype0
?
Adam/HidA/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*#
shared_nameAdam/HidA/kernel/m
|
&Adam/HidA/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidA/kernel/m*!
_output_shapes
:???*
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
?F?*$
shared_nameAdam/HidA3/kernel/m
}
'Adam/HidA3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidA3/kernel/m* 
_output_shapes
:
?F?*
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
??*$
shared_nameAdam/HidA4/kernel/m
}
'Adam/HidA4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidA4/kernel/m* 
_output_shapes
:
??*
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
Adam/HidB/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ȓ?*#
shared_nameAdam/HidB/kernel/m
|
&Adam/HidB/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidB/kernel/m*!
_output_shapes
:ȓ?*
dtype0
y
Adam/HidB/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/HidB/bias/m
r
$Adam/HidB/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidB/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/HidB2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*$
shared_nameAdam/HidB2/kernel/m
~
'Adam/HidB2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidB2/kernel/m*!
_output_shapes
:???*
dtype0
{
Adam/HidB2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidB2/bias/m
t
%Adam/HidB2/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidB2/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/HidB3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?F?*$
shared_nameAdam/HidB3/kernel/m
}
'Adam/HidB3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidB3/kernel/m* 
_output_shapes
:
?F?*
dtype0
{
Adam/HidB3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidB3/bias/m
t
%Adam/HidB3/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidB3/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/HidC/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*#
shared_nameAdam/HidC/kernel/m
|
&Adam/HidC/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidC/kernel/m*!
_output_shapes
:???*
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
Adam/HidD1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*$
shared_nameAdam/HidD1/kernel/m
~
'Adam/HidD1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidD1/kernel/m*!
_output_shapes
:???*
dtype0
{
Adam/HidD1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidD1/bias/m
t
%Adam/HidD1/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidD1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/HidE1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*$
shared_nameAdam/HidE1/kernel/m
~
'Adam/HidE1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidE1/kernel/m*!
_output_shapes
:???*
dtype0
{
Adam/HidE1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidE1/bias/m
t
%Adam/HidE1/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidE1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/Hid1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_nameAdam/Hid1/kernel/m
{
&Adam/Hid1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Hid1/kernel/m* 
_output_shapes
:
??*
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
dtype0*
shape:
??*%
shared_nameAdam/output/kernel/m

(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m* 
_output_shapes
:
??*
dtype0
}
Adam/output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_nameAdam/output/bias/m
v
&Adam/output/bias/m/Read/ReadVariableOpReadVariableOpAdam/output/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/ConA/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/ConA/kernel/v
?
&Adam/ConA/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConA/kernel/v*&
_output_shapes
:*
dtype0
x
Adam/ConA/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/ConA/bias/v
q
$Adam/ConA/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConA/bias/v*
_output_shapes
:*
dtype0
?
Adam/ConB/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/ConB/kernel/v
?
&Adam/ConB/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConB/kernel/v*&
_output_shapes
:*
dtype0
x
Adam/ConB/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/ConB/bias/v
q
$Adam/ConB/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConB/bias/v*
_output_shapes
:*
dtype0
?
Adam/ConA2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConA2/kernel/v
?
'Adam/ConA2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConA2/kernel/v*&
_output_shapes
:*
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
Adam/ConC/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*#
shared_nameAdam/ConC/kernel/v
?
&Adam/ConC/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConC/kernel/v*&
_output_shapes
:	*
dtype0
x
Adam/ConC/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/ConC/bias/v
q
$Adam/ConC/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConC/bias/v*
_output_shapes
:*
dtype0
?
Adam/ConA3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*$
shared_nameAdam/ConA3/kernel/v
?
'Adam/ConA3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConA3/kernel/v*&
_output_shapes
:(*
dtype0
z
Adam/ConA3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*"
shared_nameAdam/ConA3/bias/v
s
%Adam/ConA3/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConA3/bias/v*
_output_shapes
:(*
dtype0
?
Adam/ConB2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConB2/kernel/v
?
'Adam/ConB2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConB2/kernel/v*&
_output_shapes
:*
dtype0
z
Adam/ConB2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/ConB2/bias/v
s
%Adam/ConB2/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConB2/bias/v*
_output_shapes
:*
dtype0
?
Adam/ConA4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:((*$
shared_nameAdam/ConA4/kernel/v
?
'Adam/ConA4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConA4/kernel/v*&
_output_shapes
:((*
dtype0
z
Adam/ConA4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*"
shared_nameAdam/ConA4/bias/v
s
%Adam/ConA4/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConA4/bias/v*
_output_shapes
:(*
dtype0
?
Adam/ConB3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*$
shared_nameAdam/ConB3/kernel/v
?
'Adam/ConB3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConB3/kernel/v*&
_output_shapes
:(*
dtype0
z
Adam/ConB3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*"
shared_nameAdam/ConB3/bias/v
s
%Adam/ConB3/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConB3/bias/v*
_output_shapes
:(*
dtype0
?
Adam/ConC2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConC2/kernel/v
?
'Adam/ConC2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConC2/kernel/v*&
_output_shapes
:*
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
Adam/ConD/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*#
shared_nameAdam/ConD/kernel/v
?
&Adam/ConD/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConD/kernel/v*&
_output_shapes
:	*
dtype0
x
Adam/ConD/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/ConD/bias/v
q
$Adam/ConD/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConD/bias/v*
_output_shapes
:*
dtype0
?
Adam/ConE/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/ConE/kernel/v
?
&Adam/ConE/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConE/kernel/v*&
_output_shapes
: *
dtype0
x
Adam/ConE/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/ConE/bias/v
q
$Adam/ConE/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConE/bias/v*
_output_shapes
: *
dtype0
?
Adam/HidA/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*#
shared_nameAdam/HidA/kernel/v
|
&Adam/HidA/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidA/kernel/v*!
_output_shapes
:???*
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
?F?*$
shared_nameAdam/HidA3/kernel/v
}
'Adam/HidA3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidA3/kernel/v* 
_output_shapes
:
?F?*
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
??*$
shared_nameAdam/HidA4/kernel/v
}
'Adam/HidA4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidA4/kernel/v* 
_output_shapes
:
??*
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
Adam/HidB/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ȓ?*#
shared_nameAdam/HidB/kernel/v
|
&Adam/HidB/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidB/kernel/v*!
_output_shapes
:ȓ?*
dtype0
y
Adam/HidB/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/HidB/bias/v
r
$Adam/HidB/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidB/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/HidB2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*$
shared_nameAdam/HidB2/kernel/v
~
'Adam/HidB2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidB2/kernel/v*!
_output_shapes
:???*
dtype0
{
Adam/HidB2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidB2/bias/v
t
%Adam/HidB2/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidB2/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/HidB3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?F?*$
shared_nameAdam/HidB3/kernel/v
}
'Adam/HidB3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidB3/kernel/v* 
_output_shapes
:
?F?*
dtype0
{
Adam/HidB3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidB3/bias/v
t
%Adam/HidB3/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidB3/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/HidC/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*#
shared_nameAdam/HidC/kernel/v
|
&Adam/HidC/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidC/kernel/v*!
_output_shapes
:???*
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
Adam/HidD1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*$
shared_nameAdam/HidD1/kernel/v
~
'Adam/HidD1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidD1/kernel/v*!
_output_shapes
:???*
dtype0
{
Adam/HidD1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidD1/bias/v
t
%Adam/HidD1/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidD1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/HidE1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*$
shared_nameAdam/HidE1/kernel/v
~
'Adam/HidE1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidE1/kernel/v*!
_output_shapes
:???*
dtype0
{
Adam/HidE1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidE1/bias/v
t
%Adam/HidE1/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidE1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/Hid1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_nameAdam/Hid1/kernel/v
{
&Adam/Hid1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Hid1/kernel/v* 
_output_shapes
:
??*
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
dtype0*
shape:
??*%
shared_nameAdam/output/kernel/v

(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v* 
_output_shapes
:
??*
dtype0
}
Adam/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_nameAdam/output/bias/v
v
&Adam/output/bias/v/Read/ReadVariableOpReadVariableOpAdam/output/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*˺
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
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
layer_with_weights-9
layer-19
layer_with_weights-10
layer-20
layer-21
layer-22
layer-23
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
.layer_with_weights-11
.layer-45
/layer_with_weights-12
/layer-46
0layer_with_weights-13
0layer-47
1layer_with_weights-14
1layer-48
2layer_with_weights-15
2layer-49
3layer_with_weights-16
3layer-50
4layer_with_weights-17
4layer-51
5layer_with_weights-18
5layer-52
6layer_with_weights-19
6layer-53
7layer_with_weights-20
7layer-54
8layer_with_weights-21
8layer-55
9layer-56
:layer_with_weights-22
:layer-57
;layer_with_weights-23
;layer-58
<layer_with_weights-24
<layer-59
=	optimizer
>regularization_losses
?trainable_variables
@	variables
A	keras_api
B
signatures
 
h

Ckernel
Dbias
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
R
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
R
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
h

Qkernel
Rbias
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
h

Wkernel
Xbias
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
R
]regularization_losses
^trainable_variables
_	variables
`	keras_api
R
aregularization_losses
btrainable_variables
c	variables
d	keras_api
R
eregularization_losses
ftrainable_variables
g	variables
h	keras_api
h

ikernel
jbias
kregularization_losses
ltrainable_variables
m	variables
n	keras_api
h

okernel
pbias
qregularization_losses
rtrainable_variables
s	variables
t	keras_api
h

ukernel
vbias
wregularization_losses
xtrainable_variables
y	variables
z	keras_api
R
{regularization_losses
|trainable_variables
}	variables
~	keras_api
U
regularization_losses
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
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
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
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_rateCm?Dm?Qm?Rm?Wm?Xm?im?jm?om?pm?um?vm?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?Cv?Dv?Qv?Rv?Wv?Xv?iv?jv?ov?pv?uv?vv?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?
 
?
C0
D1
Q2
R3
W4
X5
i6
j7
o8
p9
u10
v11
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
?22
?23
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
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?
C0
D1
Q2
R3
W4
X5
i6
j7
o8
p9
u10
v11
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
?22
?23
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
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?
?layer_metrics
 ?layer_regularization_losses
>regularization_losses
?trainable_variables
@	variables
?metrics
?layers
?non_trainable_variables
 
WU
VARIABLE_VALUEConA/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	ConA/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

C0
D1

C0
D1
?
?layer_metrics
 ?layer_regularization_losses
Eregularization_losses
Ftrainable_variables
G	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
Iregularization_losses
Jtrainable_variables
K	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
Mregularization_losses
Ntrainable_variables
O	variables
?metrics
?layers
?non_trainable_variables
WU
VARIABLE_VALUEConB/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	ConB/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Q0
R1

Q0
R1
?
?layer_metrics
 ?layer_regularization_losses
Sregularization_losses
Ttrainable_variables
U	variables
?metrics
?layers
?non_trainable_variables
XV
VARIABLE_VALUEConA2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
ConA2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

W0
X1

W0
X1
?
?layer_metrics
 ?layer_regularization_losses
Yregularization_losses
Ztrainable_variables
[	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
]regularization_losses
^trainable_variables
_	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
aregularization_losses
btrainable_variables
c	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
eregularization_losses
ftrainable_variables
g	variables
?metrics
?layers
?non_trainable_variables
WU
VARIABLE_VALUEConC/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	ConC/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

i0
j1

i0
j1
?
?layer_metrics
 ?layer_regularization_losses
kregularization_losses
ltrainable_variables
m	variables
?metrics
?layers
?non_trainable_variables
XV
VARIABLE_VALUEConA3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
ConA3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

o0
p1

o0
p1
?
?layer_metrics
 ?layer_regularization_losses
qregularization_losses
rtrainable_variables
s	variables
?metrics
?layers
?non_trainable_variables
XV
VARIABLE_VALUEConB2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
ConB2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

u0
v1

u0
v1
?
?layer_metrics
 ?layer_regularization_losses
wregularization_losses
xtrainable_variables
y	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
{regularization_losses
|trainable_variables
}	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
XV
VARIABLE_VALUEConA4/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
ConA4/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
XV
VARIABLE_VALUEConB3/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
ConB3/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
XV
VARIABLE_VALUEConC2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
ConC2/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
WU
VARIABLE_VALUEConD/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	ConD/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
XV
VARIABLE_VALUEConE/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	ConE/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
XV
VARIABLE_VALUEHidA/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	HidA/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
YW
VARIABLE_VALUEHidA2/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidA2/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
YW
VARIABLE_VALUEHidA3/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidA3/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
YW
VARIABLE_VALUEHidA4/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidA4/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
XV
VARIABLE_VALUEHidB/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	HidB/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
YW
VARIABLE_VALUEHidB2/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidB2/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
YW
VARIABLE_VALUEHidB3/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidB3/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
XV
VARIABLE_VALUEHidC/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	HidC/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
YW
VARIABLE_VALUEHidC2/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidC2/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
YW
VARIABLE_VALUEHidD1/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidD1/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
YW
VARIABLE_VALUEHidE1/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidE1/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
XV
VARIABLE_VALUEHid1/kernel7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	Hid1/bias5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
XV
VARIABLE_VALUEHid2/kernel7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	Hid2/bias5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
ZX
VARIABLE_VALUEoutput/kernel7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEoutput/bias5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
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
?0
?1
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
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
zx
VARIABLE_VALUEAdam/ConA/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConA/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConB/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConB/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConA2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConA2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConC/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConC/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConA3/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConA3/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConB2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConB2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConA4/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConA4/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConB3/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConB3/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConC2/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConC2/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConD/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConD/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConE/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConE/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/HidA/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/HidA/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidA2/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidA2/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidA3/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidA3/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidA4/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidA4/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/HidB/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/HidB/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidB2/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidB2/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidB3/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidB3/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/HidC/kernel/mSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/HidC/bias/mQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidC2/kernel/mSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidC2/bias/mQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidD1/kernel/mSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidD1/bias/mQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidE1/kernel/mSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidE1/bias/mQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/Hid1/kernel/mSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Hid1/bias/mQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/Hid2/kernel/mSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Hid2/bias/mQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/output/kernel/mSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/output/bias/mQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConA/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConA/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConB/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConB/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConA2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConA2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConC/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConC/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConA3/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConA3/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConB2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConB2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConA4/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConA4/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConB3/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConB3/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConC2/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConC2/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConD/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConD/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConE/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConE/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/HidA/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/HidA/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidA2/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidA2/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidA3/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidA3/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidA4/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidA4/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/HidB/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/HidB/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidB2/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidB2/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidB3/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidB3/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/HidC/kernel/vSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/HidC/bias/vQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidC2/kernel/vSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidC2/bias/vQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidD1/kernel/vSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidD1/bias/vQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidE1/kernel/vSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidE1/bias/vQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/Hid1/kernel/vSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Hid1/bias/vQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/Hid2/kernel/vSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Hid2/bias/vQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/output/kernel/vSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/output/bias/vQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_inputPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputConA/kernel	ConA/biasConB/kernel	ConB/biasConA2/kernel
ConA2/biasConC/kernel	ConC/biasConB2/kernel
ConB2/biasConA3/kernel
ConA3/biasConE/kernel	ConE/biasConD/kernel	ConD/biasConC2/kernel
ConC2/biasConB3/kernel
ConB3/biasConA4/kernel
ConA4/biasHidA/kernel	HidA/biasHidA2/kernel
HidA2/biasHidA3/kernel
HidA3/biasHidA4/kernel
HidA4/biasHidB/kernel	HidB/biasHidB2/kernel
HidB2/biasHidB3/kernel
HidB3/biasHidC/kernel	HidC/biasHidC2/kernel
HidC2/biasHidD1/kernel
HidD1/biasHidE1/kernel
HidE1/biasHid1/kernel	Hid1/biasHid2/kernel	Hid2/biasoutput/kerneloutput/bias*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_193372
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?1
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameConA/kernel/Read/ReadVariableOpConA/bias/Read/ReadVariableOpConB/kernel/Read/ReadVariableOpConB/bias/Read/ReadVariableOp ConA2/kernel/Read/ReadVariableOpConA2/bias/Read/ReadVariableOpConC/kernel/Read/ReadVariableOpConC/bias/Read/ReadVariableOp ConA3/kernel/Read/ReadVariableOpConA3/bias/Read/ReadVariableOp ConB2/kernel/Read/ReadVariableOpConB2/bias/Read/ReadVariableOp ConA4/kernel/Read/ReadVariableOpConA4/bias/Read/ReadVariableOp ConB3/kernel/Read/ReadVariableOpConB3/bias/Read/ReadVariableOp ConC2/kernel/Read/ReadVariableOpConC2/bias/Read/ReadVariableOpConD/kernel/Read/ReadVariableOpConD/bias/Read/ReadVariableOpConE/kernel/Read/ReadVariableOpConE/bias/Read/ReadVariableOpHidA/kernel/Read/ReadVariableOpHidA/bias/Read/ReadVariableOp HidA2/kernel/Read/ReadVariableOpHidA2/bias/Read/ReadVariableOp HidA3/kernel/Read/ReadVariableOpHidA3/bias/Read/ReadVariableOp HidA4/kernel/Read/ReadVariableOpHidA4/bias/Read/ReadVariableOpHidB/kernel/Read/ReadVariableOpHidB/bias/Read/ReadVariableOp HidB2/kernel/Read/ReadVariableOpHidB2/bias/Read/ReadVariableOp HidB3/kernel/Read/ReadVariableOpHidB3/bias/Read/ReadVariableOpHidC/kernel/Read/ReadVariableOpHidC/bias/Read/ReadVariableOp HidC2/kernel/Read/ReadVariableOpHidC2/bias/Read/ReadVariableOp HidD1/kernel/Read/ReadVariableOpHidD1/bias/Read/ReadVariableOp HidE1/kernel/Read/ReadVariableOpHidE1/bias/Read/ReadVariableOpHid1/kernel/Read/ReadVariableOpHid1/bias/Read/ReadVariableOpHid2/kernel/Read/ReadVariableOpHid2/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp&Adam/ConA/kernel/m/Read/ReadVariableOp$Adam/ConA/bias/m/Read/ReadVariableOp&Adam/ConB/kernel/m/Read/ReadVariableOp$Adam/ConB/bias/m/Read/ReadVariableOp'Adam/ConA2/kernel/m/Read/ReadVariableOp%Adam/ConA2/bias/m/Read/ReadVariableOp&Adam/ConC/kernel/m/Read/ReadVariableOp$Adam/ConC/bias/m/Read/ReadVariableOp'Adam/ConA3/kernel/m/Read/ReadVariableOp%Adam/ConA3/bias/m/Read/ReadVariableOp'Adam/ConB2/kernel/m/Read/ReadVariableOp%Adam/ConB2/bias/m/Read/ReadVariableOp'Adam/ConA4/kernel/m/Read/ReadVariableOp%Adam/ConA4/bias/m/Read/ReadVariableOp'Adam/ConB3/kernel/m/Read/ReadVariableOp%Adam/ConB3/bias/m/Read/ReadVariableOp'Adam/ConC2/kernel/m/Read/ReadVariableOp%Adam/ConC2/bias/m/Read/ReadVariableOp&Adam/ConD/kernel/m/Read/ReadVariableOp$Adam/ConD/bias/m/Read/ReadVariableOp&Adam/ConE/kernel/m/Read/ReadVariableOp$Adam/ConE/bias/m/Read/ReadVariableOp&Adam/HidA/kernel/m/Read/ReadVariableOp$Adam/HidA/bias/m/Read/ReadVariableOp'Adam/HidA2/kernel/m/Read/ReadVariableOp%Adam/HidA2/bias/m/Read/ReadVariableOp'Adam/HidA3/kernel/m/Read/ReadVariableOp%Adam/HidA3/bias/m/Read/ReadVariableOp'Adam/HidA4/kernel/m/Read/ReadVariableOp%Adam/HidA4/bias/m/Read/ReadVariableOp&Adam/HidB/kernel/m/Read/ReadVariableOp$Adam/HidB/bias/m/Read/ReadVariableOp'Adam/HidB2/kernel/m/Read/ReadVariableOp%Adam/HidB2/bias/m/Read/ReadVariableOp'Adam/HidB3/kernel/m/Read/ReadVariableOp%Adam/HidB3/bias/m/Read/ReadVariableOp&Adam/HidC/kernel/m/Read/ReadVariableOp$Adam/HidC/bias/m/Read/ReadVariableOp'Adam/HidC2/kernel/m/Read/ReadVariableOp%Adam/HidC2/bias/m/Read/ReadVariableOp'Adam/HidD1/kernel/m/Read/ReadVariableOp%Adam/HidD1/bias/m/Read/ReadVariableOp'Adam/HidE1/kernel/m/Read/ReadVariableOp%Adam/HidE1/bias/m/Read/ReadVariableOp&Adam/Hid1/kernel/m/Read/ReadVariableOp$Adam/Hid1/bias/m/Read/ReadVariableOp&Adam/Hid2/kernel/m/Read/ReadVariableOp$Adam/Hid2/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp&Adam/ConA/kernel/v/Read/ReadVariableOp$Adam/ConA/bias/v/Read/ReadVariableOp&Adam/ConB/kernel/v/Read/ReadVariableOp$Adam/ConB/bias/v/Read/ReadVariableOp'Adam/ConA2/kernel/v/Read/ReadVariableOp%Adam/ConA2/bias/v/Read/ReadVariableOp&Adam/ConC/kernel/v/Read/ReadVariableOp$Adam/ConC/bias/v/Read/ReadVariableOp'Adam/ConA3/kernel/v/Read/ReadVariableOp%Adam/ConA3/bias/v/Read/ReadVariableOp'Adam/ConB2/kernel/v/Read/ReadVariableOp%Adam/ConB2/bias/v/Read/ReadVariableOp'Adam/ConA4/kernel/v/Read/ReadVariableOp%Adam/ConA4/bias/v/Read/ReadVariableOp'Adam/ConB3/kernel/v/Read/ReadVariableOp%Adam/ConB3/bias/v/Read/ReadVariableOp'Adam/ConC2/kernel/v/Read/ReadVariableOp%Adam/ConC2/bias/v/Read/ReadVariableOp&Adam/ConD/kernel/v/Read/ReadVariableOp$Adam/ConD/bias/v/Read/ReadVariableOp&Adam/ConE/kernel/v/Read/ReadVariableOp$Adam/ConE/bias/v/Read/ReadVariableOp&Adam/HidA/kernel/v/Read/ReadVariableOp$Adam/HidA/bias/v/Read/ReadVariableOp'Adam/HidA2/kernel/v/Read/ReadVariableOp%Adam/HidA2/bias/v/Read/ReadVariableOp'Adam/HidA3/kernel/v/Read/ReadVariableOp%Adam/HidA3/bias/v/Read/ReadVariableOp'Adam/HidA4/kernel/v/Read/ReadVariableOp%Adam/HidA4/bias/v/Read/ReadVariableOp&Adam/HidB/kernel/v/Read/ReadVariableOp$Adam/HidB/bias/v/Read/ReadVariableOp'Adam/HidB2/kernel/v/Read/ReadVariableOp%Adam/HidB2/bias/v/Read/ReadVariableOp'Adam/HidB3/kernel/v/Read/ReadVariableOp%Adam/HidB3/bias/v/Read/ReadVariableOp&Adam/HidC/kernel/v/Read/ReadVariableOp$Adam/HidC/bias/v/Read/ReadVariableOp'Adam/HidC2/kernel/v/Read/ReadVariableOp%Adam/HidC2/bias/v/Read/ReadVariableOp'Adam/HidD1/kernel/v/Read/ReadVariableOp%Adam/HidD1/bias/v/Read/ReadVariableOp'Adam/HidE1/kernel/v/Read/ReadVariableOp%Adam/HidE1/bias/v/Read/ReadVariableOp&Adam/Hid1/kernel/v/Read/ReadVariableOp$Adam/Hid1/bias/v/Read/ReadVariableOp&Adam/Hid2/kernel/v/Read/ReadVariableOp$Adam/Hid2/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*?
Tin?
?2?	*
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
__inference__traced_save_195778
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConA/kernel	ConA/biasConB/kernel	ConB/biasConA2/kernel
ConA2/biasConC/kernel	ConC/biasConA3/kernel
ConA3/biasConB2/kernel
ConB2/biasConA4/kernel
ConA4/biasConB3/kernel
ConB3/biasConC2/kernel
ConC2/biasConD/kernel	ConD/biasConE/kernel	ConE/biasHidA/kernel	HidA/biasHidA2/kernel
HidA2/biasHidA3/kernel
HidA3/biasHidA4/kernel
HidA4/biasHidB/kernel	HidB/biasHidB2/kernel
HidB2/biasHidB3/kernel
HidB3/biasHidC/kernel	HidC/biasHidC2/kernel
HidC2/biasHidD1/kernel
HidD1/biasHidE1/kernel
HidE1/biasHid1/kernel	Hid1/biasHid2/kernel	Hid2/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/ConA/kernel/mAdam/ConA/bias/mAdam/ConB/kernel/mAdam/ConB/bias/mAdam/ConA2/kernel/mAdam/ConA2/bias/mAdam/ConC/kernel/mAdam/ConC/bias/mAdam/ConA3/kernel/mAdam/ConA3/bias/mAdam/ConB2/kernel/mAdam/ConB2/bias/mAdam/ConA4/kernel/mAdam/ConA4/bias/mAdam/ConB3/kernel/mAdam/ConB3/bias/mAdam/ConC2/kernel/mAdam/ConC2/bias/mAdam/ConD/kernel/mAdam/ConD/bias/mAdam/ConE/kernel/mAdam/ConE/bias/mAdam/HidA/kernel/mAdam/HidA/bias/mAdam/HidA2/kernel/mAdam/HidA2/bias/mAdam/HidA3/kernel/mAdam/HidA3/bias/mAdam/HidA4/kernel/mAdam/HidA4/bias/mAdam/HidB/kernel/mAdam/HidB/bias/mAdam/HidB2/kernel/mAdam/HidB2/bias/mAdam/HidB3/kernel/mAdam/HidB3/bias/mAdam/HidC/kernel/mAdam/HidC/bias/mAdam/HidC2/kernel/mAdam/HidC2/bias/mAdam/HidD1/kernel/mAdam/HidD1/bias/mAdam/HidE1/kernel/mAdam/HidE1/bias/mAdam/Hid1/kernel/mAdam/Hid1/bias/mAdam/Hid2/kernel/mAdam/Hid2/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/ConA/kernel/vAdam/ConA/bias/vAdam/ConB/kernel/vAdam/ConB/bias/vAdam/ConA2/kernel/vAdam/ConA2/bias/vAdam/ConC/kernel/vAdam/ConC/bias/vAdam/ConA3/kernel/vAdam/ConA3/bias/vAdam/ConB2/kernel/vAdam/ConB2/bias/vAdam/ConA4/kernel/vAdam/ConA4/bias/vAdam/ConB3/kernel/vAdam/ConB3/bias/vAdam/ConC2/kernel/vAdam/ConC2/bias/vAdam/ConD/kernel/vAdam/ConD/bias/vAdam/ConE/kernel/vAdam/ConE/bias/vAdam/HidA/kernel/vAdam/HidA/bias/vAdam/HidA2/kernel/vAdam/HidA2/bias/vAdam/HidA3/kernel/vAdam/HidA3/bias/vAdam/HidA4/kernel/vAdam/HidA4/bias/vAdam/HidB/kernel/vAdam/HidB/bias/vAdam/HidB2/kernel/vAdam/HidB2/bias/vAdam/HidB3/kernel/vAdam/HidB3/bias/vAdam/HidC/kernel/vAdam/HidC/bias/vAdam/HidC2/kernel/vAdam/HidC2/bias/vAdam/HidD1/kernel/vAdam/HidD1/bias/vAdam/HidE1/kernel/vAdam/HidE1/bias/vAdam/Hid1/kernel/vAdam/Hid1/bias/vAdam/Hid2/kernel/vAdam/Hid2/bias/vAdam/output/kernel/vAdam/output/bias/v*?
Tin?
?2?*
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
"__inference__traced_restore_196265??%
?
?
'__inference_output_layer_call_fn_195278

inputs
unknown:
??
	unknown_0:	?
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
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1917052
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

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
?
a
E__inference_flatten_9_layer_call_and_return_conditional_losses_194951

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????? 2
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
:?????????G=:W S
/
_output_shapes
:?????????G=
 
_user_specified_nameinputs
?
B
&__inference_PoolE_layer_call_fn_194625

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
A__inference_PoolE_layer_call_and_return_conditional_losses_1910172
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
@__inference_HidC_layer_call_and_return_conditional_losses_191585

inputs3
matmul_readvariableop_resource:???.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
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
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?"
A__inference_model_layer_call_and_return_conditional_losses_193597

inputs=
#cona_conv2d_readvariableop_resource:2
$cona_biasadd_readvariableop_resource:=
#conb_conv2d_readvariableop_resource:2
$conb_biasadd_readvariableop_resource:>
$cona2_conv2d_readvariableop_resource:3
%cona2_biasadd_readvariableop_resource:=
#conc_conv2d_readvariableop_resource:	2
$conc_biasadd_readvariableop_resource:>
$conb2_conv2d_readvariableop_resource:3
%conb2_biasadd_readvariableop_resource:>
$cona3_conv2d_readvariableop_resource:(3
%cona3_biasadd_readvariableop_resource:(=
#cone_conv2d_readvariableop_resource: 2
$cone_biasadd_readvariableop_resource: =
#cond_conv2d_readvariableop_resource:	2
$cond_biasadd_readvariableop_resource:>
$conc2_conv2d_readvariableop_resource:3
%conc2_biasadd_readvariableop_resource:>
$conb3_conv2d_readvariableop_resource:(3
%conb3_biasadd_readvariableop_resource:(>
$cona4_conv2d_readvariableop_resource:((3
%cona4_biasadd_readvariableop_resource:(8
#hida_matmul_readvariableop_resource:???3
$hida_biasadd_readvariableop_resource:	?9
$hida2_matmul_readvariableop_resource:???4
%hida2_biasadd_readvariableop_resource:	?8
$hida3_matmul_readvariableop_resource:
?F?4
%hida3_biasadd_readvariableop_resource:	?8
$hida4_matmul_readvariableop_resource:
??4
%hida4_biasadd_readvariableop_resource:	?8
#hidb_matmul_readvariableop_resource:ȓ?3
$hidb_biasadd_readvariableop_resource:	?9
$hidb2_matmul_readvariableop_resource:???4
%hidb2_biasadd_readvariableop_resource:	?8
$hidb3_matmul_readvariableop_resource:
?F?4
%hidb3_biasadd_readvariableop_resource:	?8
#hidc_matmul_readvariableop_resource:???3
$hidc_biasadd_readvariableop_resource:	?9
$hidc2_matmul_readvariableop_resource:???4
%hidc2_biasadd_readvariableop_resource:	?9
$hidd1_matmul_readvariableop_resource:???4
%hidd1_biasadd_readvariableop_resource:	?9
$hide1_matmul_readvariableop_resource:???4
%hide1_biasadd_readvariableop_resource:	?7
#hid1_matmul_readvariableop_resource:
??3
$hid1_biasadd_readvariableop_resource:	?7
#hid2_matmul_readvariableop_resource:
??3
$hid2_biasadd_readvariableop_resource:	?9
%output_matmul_readvariableop_resource:
??5
&output_biasadd_readvariableop_resource:	?
identity??ConA/BiasAdd/ReadVariableOp?ConA/Conv2D/ReadVariableOp?ConA2/BiasAdd/ReadVariableOp?ConA2/Conv2D/ReadVariableOp?ConA3/BiasAdd/ReadVariableOp?ConA3/Conv2D/ReadVariableOp?ConA4/BiasAdd/ReadVariableOp?ConA4/Conv2D/ReadVariableOp?ConB/BiasAdd/ReadVariableOp?ConB/Conv2D/ReadVariableOp?ConB2/BiasAdd/ReadVariableOp?ConB2/Conv2D/ReadVariableOp?ConB3/BiasAdd/ReadVariableOp?ConB3/Conv2D/ReadVariableOp?ConC/BiasAdd/ReadVariableOp?ConC/Conv2D/ReadVariableOp?ConC2/BiasAdd/ReadVariableOp?ConC2/Conv2D/ReadVariableOp?ConD/BiasAdd/ReadVariableOp?ConD/Conv2D/ReadVariableOp?ConE/BiasAdd/ReadVariableOp?ConE/Conv2D/ReadVariableOp?Hid1/BiasAdd/ReadVariableOp?Hid1/MatMul/ReadVariableOp?Hid2/BiasAdd/ReadVariableOp?Hid2/MatMul/ReadVariableOp?HidA/BiasAdd/ReadVariableOp?HidA/MatMul/ReadVariableOp?HidA2/BiasAdd/ReadVariableOp?HidA2/MatMul/ReadVariableOp?HidA3/BiasAdd/ReadVariableOp?HidA3/MatMul/ReadVariableOp?HidA4/BiasAdd/ReadVariableOp?HidA4/MatMul/ReadVariableOp?HidB/BiasAdd/ReadVariableOp?HidB/MatMul/ReadVariableOp?HidB2/BiasAdd/ReadVariableOp?HidB2/MatMul/ReadVariableOp?HidB3/BiasAdd/ReadVariableOp?HidB3/MatMul/ReadVariableOp?HidC/BiasAdd/ReadVariableOp?HidC/MatMul/ReadVariableOp?HidC2/BiasAdd/ReadVariableOp?HidC2/MatMul/ReadVariableOp?HidD1/BiasAdd/ReadVariableOp?HidD1/MatMul/ReadVariableOp?HidE1/BiasAdd/ReadVariableOp?HidE1/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
ConA/Conv2D/ReadVariableOpReadVariableOp#cona_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConA/Conv2D/ReadVariableOp?
ConA/Conv2DConv2Dinputs"ConA/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
ConA/Conv2D?
ConA/BiasAdd/ReadVariableOpReadVariableOp$cona_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConA/BiasAdd/ReadVariableOp?
ConA/BiasAddBiasAddConA/Conv2D:output:0#ConA/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
ConA/BiasAddq
	ConA/ReluReluConA/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
	ConA/Relu?
PoolA/MaxPoolMaxPoolConA/Relu:activations:0*/
_output_shapes
:?????????I?*
ksize
*
paddingVALID*
strides
2
PoolA/MaxPool?
ConB/Conv2D/ReadVariableOpReadVariableOp#conb_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConB/Conv2D/ReadVariableOp?
ConB/Conv2DConv2Dinputs"ConB/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
ConB/Conv2D?
ConB/BiasAdd/ReadVariableOpReadVariableOp$conb_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConB/BiasAdd/ReadVariableOp?
ConB/BiasAddBiasAddConB/Conv2D:output:0#ConB/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
ConB/BiasAddq
	ConB/ReluReluConB/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
	ConB/Relu?
dropout/IdentityIdentityPoolA/MaxPool:output:0*
T0*/
_output_shapes
:?????????I?2
dropout/Identity?
PoolB/MaxPoolMaxPoolConB/Relu:activations:0*/
_output_shapes
:?????????H?*
ksize
*
paddingVALID*
strides
2
PoolB/MaxPool?
ConA2/Conv2D/ReadVariableOpReadVariableOp$cona2_conv2d_readvariableop_resource*&
_output_shapes
:*
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
ConC/Conv2D/ReadVariableOpReadVariableOp#conc_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
ConC/Conv2D/ReadVariableOp?
ConC/Conv2DConv2Dinputs"ConC/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
ConC/Conv2D?
ConC/BiasAdd/ReadVariableOpReadVariableOp$conc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConC/BiasAdd/ReadVariableOp?
ConC/BiasAddBiasAddConC/Conv2D:output:0#ConC/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
ConC/BiasAddq
	ConC/ReluReluConC/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
	ConC/Relu?
dropout_1/IdentityIdentityPoolB/MaxPool:output:0*
T0*/
_output_shapes
:?????????H?2
dropout_1/Identity?
PoolA2/MaxPoolMaxPoolConA2/Relu:activations:0*/
_output_shapes
:?????????#*
ksize
*
paddingVALID*
strides
2
PoolA2/MaxPool?
PoolC/MaxPoolMaxPoolConC/Relu:activations:0*/
_output_shapes
:?????????G>*
ksize
*
paddingVALID*
strides
2
PoolC/MaxPool?
ConB2/Conv2D/ReadVariableOpReadVariableOp$conb2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConB2/Conv2D/ReadVariableOp?
ConB2/Conv2DConv2Ddropout_1/Identity:output:0#ConB2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????F=*
paddingVALID*
strides
2
ConB2/Conv2D?
ConB2/BiasAdd/ReadVariableOpReadVariableOp%conb2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConB2/BiasAdd/ReadVariableOp?
ConB2/BiasAddBiasAddConB2/Conv2D:output:0$ConB2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????F=2
ConB2/BiasAddr

ConB2/ReluReluConB2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????F=2

ConB2/Relu?
ConA3/Conv2D/ReadVariableOpReadVariableOp$cona3_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
ConA3/Conv2D/ReadVariableOp?
ConA3/Conv2DConv2DPoolA2/MaxPool:output:0#ConA3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!(*
paddingVALID*
strides
2
ConA3/Conv2D?
ConA3/BiasAdd/ReadVariableOpReadVariableOp%cona3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
ConA3/BiasAdd/ReadVariableOp?
ConA3/BiasAddBiasAddConA3/Conv2D:output:0$ConA3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!(2
ConA3/BiasAddr

ConA3/ReluReluConA3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????!(2

ConA3/Relu?
dropout_2/IdentityIdentityPoolC/MaxPool:output:0*
T0*/
_output_shapes
:?????????G>2
dropout_2/Identity?
PoolB2/MaxPoolMaxPoolConB2/Relu:activations:0*/
_output_shapes
:?????????#*
ksize
*
paddingVALID*
strides
2
PoolB2/MaxPool?
PoolA3/MaxPoolMaxPoolConA3/Relu:activations:0*/
_output_shapes
:?????????(*
ksize
*
paddingVALID*
strides
2
PoolA3/MaxPool?
ConE/Conv2D/ReadVariableOpReadVariableOp#cone_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
ConE/Conv2D/ReadVariableOp?
ConE/Conv2DConv2Dinputs"ConE/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
2
ConE/Conv2D?
ConE/BiasAdd/ReadVariableOpReadVariableOp$cone_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
ConE/BiasAdd/ReadVariableOp?
ConE/BiasAddBiasAddConE/Conv2D:output:0#ConE/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
ConE/BiasAddq
	ConE/ReluReluConE/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
	ConE/Relu?
ConD/Conv2D/ReadVariableOpReadVariableOp#cond_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
ConD/Conv2D/ReadVariableOp?
ConD/Conv2DConv2Dinputs"ConD/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
ConD/Conv2D?
ConD/BiasAdd/ReadVariableOpReadVariableOp$cond_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConD/BiasAdd/ReadVariableOp?
ConD/BiasAddBiasAddConD/Conv2D:output:0#ConD/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
ConD/BiasAddq
	ConD/ReluReluConD/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
	ConD/Relu?
ConC2/Conv2D/ReadVariableOpReadVariableOp$conc2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConC2/Conv2D/ReadVariableOp?
ConC2/Conv2DConv2Ddropout_2/Identity:output:0#ConC2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????E<*
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
:?????????E<2
ConC2/BiasAddr

ConC2/ReluReluConC2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????E<2

ConC2/Relu?
ConB3/Conv2D/ReadVariableOpReadVariableOp$conb3_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
ConB3/Conv2D/ReadVariableOp?
ConB3/Conv2DConv2DPoolB2/MaxPool:output:0#ConB3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!(*
paddingVALID*
strides
2
ConB3/Conv2D?
ConB3/BiasAdd/ReadVariableOpReadVariableOp%conb3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
ConB3/BiasAdd/ReadVariableOp?
ConB3/BiasAddBiasAddConB3/Conv2D:output:0$ConB3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!(2
ConB3/BiasAddr

ConB3/ReluReluConB3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????!(2

ConB3/Relu?
ConA4/Conv2D/ReadVariableOpReadVariableOp$cona4_conv2d_readvariableop_resource*&
_output_shapes
:((*
dtype02
ConA4/Conv2D/ReadVariableOp?
ConA4/Conv2DConv2DPoolA3/MaxPool:output:0#ConA4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingVALID*
strides
2
ConA4/Conv2D?
ConA4/BiasAdd/ReadVariableOpReadVariableOp%cona4_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
ConA4/BiasAdd/ReadVariableOp?
ConA4/BiasAddBiasAddConA4/Conv2D:output:0$ConA4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2
ConA4/BiasAddr

ConA4/ReluReluConA4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(2

ConA4/Relu?
PoolE/MaxPoolMaxPoolConE/Relu:activations:0*/
_output_shapes
:?????????D; *
ksize
*
paddingVALID*
strides
2
PoolE/MaxPool?
PoolD/MaxPoolMaxPoolConD/Relu:activations:0*/
_output_shapes
:?????????G=*
ksize
*
paddingVALID*
strides
2
PoolD/MaxPool?
PoolC2/MaxPoolMaxPoolConC2/Relu:activations:0*/
_output_shapes
:?????????"*
ksize
*
paddingVALID*
strides
2
PoolC2/MaxPool?
PoolB3/MaxPoolMaxPoolConB3/Relu:activations:0*/
_output_shapes
:?????????(*
ksize
*
paddingVALID*
strides
2
PoolB3/MaxPool?
PoolA4/MaxPoolMaxPoolConA4/Relu:activations:0*/
_output_shapes
:?????????(*
ksize
*
paddingVALID*
strides
2
PoolA4/MaxPool?
dropout_4/IdentityIdentityPoolE/MaxPool:output:0*
T0*/
_output_shapes
:?????????D; 2
dropout_4/Identity?
dropout_3/IdentityIdentityPoolD/MaxPool:output:0*
T0*/
_output_shapes
:?????????G=2
dropout_3/Identity?
dropout_10/IdentityIdentityPoolC2/MaxPool:output:0*
T0*/
_output_shapes
:?????????"2
dropout_10/Identity?
dropout_9/IdentityIdentityPoolB3/MaxPool:output:0*
T0*/
_output_shapes
:?????????(2
dropout_9/Identity?
dropout_8/IdentityIdentityPoolB2/MaxPool:output:0*
T0*/
_output_shapes
:?????????#2
dropout_8/Identity?
dropout_7/IdentityIdentityPoolA4/MaxPool:output:0*
T0*/
_output_shapes
:?????????(2
dropout_7/Identity?
dropout_6/IdentityIdentityPoolA3/MaxPool:output:0*
T0*/
_output_shapes
:?????????(2
dropout_6/Identity?
dropout_5/IdentityIdentityPoolA2/MaxPool:output:0*
T0*/
_output_shapes
:?????????#2
dropout_5/Identityu
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????? 2
flatten_10/Const?
flatten_10/ReshapeReshapedropout_4/Identity:output:0flatten_10/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_10/Reshapes
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????? 2
flatten_9/Const?
flatten_9/ReshapeReshapedropout_3/Identity:output:0flatten_9/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_9/Reshapes
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????O  2
flatten_8/Const?
flatten_8/ReshapeReshapedropout_10/Identity:output:0flatten_8/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_8/Reshapes
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????W 2
flatten_7/Const?
flatten_7/ReshapeReshapedropout_2/Identity:output:0flatten_7/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_7/Reshapes
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? #  2
flatten_6/Const?
flatten_6/ReshapeReshapedropout_9/Identity:output:0flatten_6/Const:output:0*
T0*(
_output_shapes
:??????????F2
flatten_6/Reshapes
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????R  2
flatten_5/Const?
flatten_5/ReshapeReshapedropout_8/Identity:output:0flatten_5/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_5/Reshapes
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????	 2
flatten_4/Const?
flatten_4/ReshapeReshapePoolB/MaxPool:output:0flatten_4/Const:output:0*
T0*)
_output_shapes
:?????????ȓ2
flatten_4/Reshapes
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_3/Const?
flatten_3/ReshapeReshapedropout_7/Identity:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_3/Reshapes
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? #  2
flatten_2/Const?
flatten_2/ReshapeReshapedropout_6/Identity:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????F2
flatten_2/Reshapes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????R  2
flatten_1/Const?
flatten_1/ReshapeReshapedropout_5/Identity:output:0flatten_1/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_1/Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"??????  2
flatten/Const?
flatten/ReshapeReshapePoolA/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten/Reshape?
HidA/MatMul/ReadVariableOpReadVariableOp#hida_matmul_readvariableop_resource*!
_output_shapes
:???*
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
?F?*
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
??*
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
HidB/MatMul/ReadVariableOpReadVariableOp#hidb_matmul_readvariableop_resource*!
_output_shapes
:ȓ?*
dtype02
HidB/MatMul/ReadVariableOp?
HidB/MatMulMatMulflatten_4/Reshape:output:0"HidB/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidB/MatMul?
HidB/BiasAdd/ReadVariableOpReadVariableOp$hidb_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidB/BiasAdd/ReadVariableOp?
HidB/BiasAddBiasAddHidB/MatMul:product:0#HidB/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidB/BiasAddh
	HidB/ReluReluHidB/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
	HidB/Relu?
HidB2/MatMul/ReadVariableOpReadVariableOp$hidb2_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
HidB2/MatMul/ReadVariableOp?
HidB2/MatMulMatMulflatten_5/Reshape:output:0#HidB2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidB2/MatMul?
HidB2/BiasAdd/ReadVariableOpReadVariableOp%hidb2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidB2/BiasAdd/ReadVariableOp?
HidB2/BiasAddBiasAddHidB2/MatMul:product:0$HidB2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidB2/BiasAddk

HidB2/ReluReluHidB2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidB2/Relu?
HidB3/MatMul/ReadVariableOpReadVariableOp$hidb3_matmul_readvariableop_resource* 
_output_shapes
:
?F?*
dtype02
HidB3/MatMul/ReadVariableOp?
HidB3/MatMulMatMulflatten_6/Reshape:output:0#HidB3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidB3/MatMul?
HidB3/BiasAdd/ReadVariableOpReadVariableOp%hidb3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidB3/BiasAdd/ReadVariableOp?
HidB3/BiasAddBiasAddHidB3/MatMul:product:0$HidB3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidB3/BiasAddk

HidB3/ReluReluHidB3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidB3/Relu?
HidC/MatMul/ReadVariableOpReadVariableOp#hidc_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
HidC/MatMul/ReadVariableOp?
HidC/MatMulMatMulflatten_7/Reshape:output:0"HidC/MatMul/ReadVariableOp:value:0*
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
HidC2/MatMulMatMulflatten_8/Reshape:output:0#HidC2/MatMul/ReadVariableOp:value:0*
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
HidD1/MatMul/ReadVariableOpReadVariableOp$hidd1_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
HidD1/MatMul/ReadVariableOp?
HidD1/MatMulMatMulflatten_9/Reshape:output:0#HidD1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidD1/MatMul?
HidD1/BiasAdd/ReadVariableOpReadVariableOp%hidd1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidD1/BiasAdd/ReadVariableOp?
HidD1/BiasAddBiasAddHidD1/MatMul:product:0$HidD1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidD1/BiasAddk

HidD1/ReluReluHidD1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidD1/Relu?
HidE1/MatMul/ReadVariableOpReadVariableOp$hide1_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
HidE1/MatMul/ReadVariableOp?
HidE1/MatMulMatMulflatten_10/Reshape:output:0#HidE1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidE1/MatMul?
HidE1/BiasAdd/ReadVariableOpReadVariableOp%hide1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidE1/BiasAdd/ReadVariableOp?
HidE1/BiasAddBiasAddHidE1/MatMul:product:0$HidE1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidE1/BiasAddk

HidE1/ReluReluHidE1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidE1/Relut
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2HidA/Relu:activations:0HidA2/Relu:activations:0HidA3/Relu:activations:0HidA4/Relu:activations:0HidB/Relu:activations:0HidB2/Relu:activations:0HidB3/Relu:activations:0HidC/Relu:activations:0HidC2/Relu:activations:0HidD1/Relu:activations:0HidE1/Relu:activations:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate/concat?
Hid1/MatMul/ReadVariableOpReadVariableOp#hid1_matmul_readvariableop_resource* 
_output_shapes
:
??*
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
_output_shapes
:
??*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMulHid2/Tanh:y:0$output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
output/BiasAddw
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
output/Softmaxt
IdentityIdentityoutput/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^ConA/BiasAdd/ReadVariableOp^ConA/Conv2D/ReadVariableOp^ConA2/BiasAdd/ReadVariableOp^ConA2/Conv2D/ReadVariableOp^ConA3/BiasAdd/ReadVariableOp^ConA3/Conv2D/ReadVariableOp^ConA4/BiasAdd/ReadVariableOp^ConA4/Conv2D/ReadVariableOp^ConB/BiasAdd/ReadVariableOp^ConB/Conv2D/ReadVariableOp^ConB2/BiasAdd/ReadVariableOp^ConB2/Conv2D/ReadVariableOp^ConB3/BiasAdd/ReadVariableOp^ConB3/Conv2D/ReadVariableOp^ConC/BiasAdd/ReadVariableOp^ConC/Conv2D/ReadVariableOp^ConC2/BiasAdd/ReadVariableOp^ConC2/Conv2D/ReadVariableOp^ConD/BiasAdd/ReadVariableOp^ConD/Conv2D/ReadVariableOp^ConE/BiasAdd/ReadVariableOp^ConE/Conv2D/ReadVariableOp^Hid1/BiasAdd/ReadVariableOp^Hid1/MatMul/ReadVariableOp^Hid2/BiasAdd/ReadVariableOp^Hid2/MatMul/ReadVariableOp^HidA/BiasAdd/ReadVariableOp^HidA/MatMul/ReadVariableOp^HidA2/BiasAdd/ReadVariableOp^HidA2/MatMul/ReadVariableOp^HidA3/BiasAdd/ReadVariableOp^HidA3/MatMul/ReadVariableOp^HidA4/BiasAdd/ReadVariableOp^HidA4/MatMul/ReadVariableOp^HidB/BiasAdd/ReadVariableOp^HidB/MatMul/ReadVariableOp^HidB2/BiasAdd/ReadVariableOp^HidB2/MatMul/ReadVariableOp^HidB3/BiasAdd/ReadVariableOp^HidB3/MatMul/ReadVariableOp^HidC/BiasAdd/ReadVariableOp^HidC/MatMul/ReadVariableOp^HidC2/BiasAdd/ReadVariableOp^HidC2/MatMul/ReadVariableOp^HidD1/BiasAdd/ReadVariableOp^HidD1/MatMul/ReadVariableOp^HidE1/BiasAdd/ReadVariableOp^HidE1/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
ConA/BiasAdd/ReadVariableOpConA/BiasAdd/ReadVariableOp28
ConA/Conv2D/ReadVariableOpConA/Conv2D/ReadVariableOp2<
ConA2/BiasAdd/ReadVariableOpConA2/BiasAdd/ReadVariableOp2:
ConA2/Conv2D/ReadVariableOpConA2/Conv2D/ReadVariableOp2<
ConA3/BiasAdd/ReadVariableOpConA3/BiasAdd/ReadVariableOp2:
ConA3/Conv2D/ReadVariableOpConA3/Conv2D/ReadVariableOp2<
ConA4/BiasAdd/ReadVariableOpConA4/BiasAdd/ReadVariableOp2:
ConA4/Conv2D/ReadVariableOpConA4/Conv2D/ReadVariableOp2:
ConB/BiasAdd/ReadVariableOpConB/BiasAdd/ReadVariableOp28
ConB/Conv2D/ReadVariableOpConB/Conv2D/ReadVariableOp2<
ConB2/BiasAdd/ReadVariableOpConB2/BiasAdd/ReadVariableOp2:
ConB2/Conv2D/ReadVariableOpConB2/Conv2D/ReadVariableOp2<
ConB3/BiasAdd/ReadVariableOpConB3/BiasAdd/ReadVariableOp2:
ConB3/Conv2D/ReadVariableOpConB3/Conv2D/ReadVariableOp2:
ConC/BiasAdd/ReadVariableOpConC/BiasAdd/ReadVariableOp28
ConC/Conv2D/ReadVariableOpConC/Conv2D/ReadVariableOp2<
ConC2/BiasAdd/ReadVariableOpConC2/BiasAdd/ReadVariableOp2:
ConC2/Conv2D/ReadVariableOpConC2/Conv2D/ReadVariableOp2:
ConD/BiasAdd/ReadVariableOpConD/BiasAdd/ReadVariableOp28
ConD/Conv2D/ReadVariableOpConD/Conv2D/ReadVariableOp2:
ConE/BiasAdd/ReadVariableOpConE/BiasAdd/ReadVariableOp28
ConE/Conv2D/ReadVariableOpConE/Conv2D/ReadVariableOp2:
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
HidB/BiasAdd/ReadVariableOpHidB/BiasAdd/ReadVariableOp28
HidB/MatMul/ReadVariableOpHidB/MatMul/ReadVariableOp2<
HidB2/BiasAdd/ReadVariableOpHidB2/BiasAdd/ReadVariableOp2:
HidB2/MatMul/ReadVariableOpHidB2/MatMul/ReadVariableOp2<
HidB3/BiasAdd/ReadVariableOpHidB3/BiasAdd/ReadVariableOp2:
HidB3/MatMul/ReadVariableOpHidB3/MatMul/ReadVariableOp2:
HidC/BiasAdd/ReadVariableOpHidC/BiasAdd/ReadVariableOp28
HidC/MatMul/ReadVariableOpHidC/MatMul/ReadVariableOp2<
HidC2/BiasAdd/ReadVariableOpHidC2/BiasAdd/ReadVariableOp2:
HidC2/MatMul/ReadVariableOpHidC2/MatMul/ReadVariableOp2<
HidD1/BiasAdd/ReadVariableOpHidD1/BiasAdd/ReadVariableOp2:
HidD1/MatMul/ReadVariableOpHidD1/MatMul/ReadVariableOp2<
HidE1/BiasAdd/ReadVariableOpHidE1/BiasAdd/ReadVariableOp2:
HidE1/MatMul/ReadVariableOpHidE1/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
D
(__inference_dropout_layer_call_fn_194171

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
:?????????I?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1910822
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????I?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????I?:W S
/
_output_shapes
:?????????I?
 
_user_specified_nameinputs
?
F
*__inference_dropout_1_layer_call_fn_194278

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
:?????????H?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1911292
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????H?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????H?:W S
/
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?
?
A__inference_HidD1_layer_call_and_return_conditional_losses_195158

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
?
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_194408

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????G>2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????G>2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????G>:W S
/
_output_shapes
:?????????G>
 
_user_specified_nameinputs
?
?
&__inference_HidC2_layer_call_fn_195147

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
A__inference_HidC2_layer_call_and_return_conditional_losses_1916022
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
A__inference_ConA3_layer_call_and_return_conditional_losses_191171

inputs8
conv2d_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!(*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!(2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????!(2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????!(2

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
?
?
A__inference_ConB2_layer_call_and_return_conditional_losses_194334

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????F=*
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
:?????????F=2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????F=2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????F=2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????H?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?
?
@__inference_ConD_layer_call_and_return_conditional_losses_194501

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
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
?
?
@__inference_ConB_layer_call_and_return_conditional_losses_191071

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????2

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
C
'__inference_PoolA4_layer_call_fn_194550

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
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA4_layer_call_and_return_conditional_losses_1913092
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
a
E__inference_flatten_6_layer_call_and_return_conditional_losses_191405

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? #  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????F2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????F2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
F
*__inference_flatten_9_layer_call_fn_194956

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
GPU2*0J 8? *N
fIRG
E__inference_flatten_9_layer_call_and_return_conditional_losses_1913812
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????G=:W S
/
_output_shapes
:?????????G=
 
_user_specified_nameinputs
?
?
@__inference_ConC_layer_call_and_return_conditional_losses_191118

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
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
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????2

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
e
F__inference_dropout_10_layer_call_and_return_conditional_losses_192172

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
:?????????"2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????"*
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
:?????????"2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????"2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????"2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????"2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????":W S
/
_output_shapes
:?????????"
 
_user_specified_nameinputs
?
d
E__inference_dropout_9_layer_call_and_return_conditional_losses_194755

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
:?????????(2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????(2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????(2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_191082

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????I?2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????I?2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????I?:W S
/
_output_shapes
:?????????I?
 
_user_specified_nameinputs
?
?
G__inference_concatenate_layer_call_and_return_conditional_losses_191658

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
	inputs_10
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:P L
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
 
_user_specified_nameinputs
?
?
&__inference_ConC2_layer_call_fn_194490

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
:?????????E<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConC2_layer_call_and_return_conditional_losses_1912412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????E<2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????G>: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????G>
 
_user_specified_nameinputs
?
c
E__inference_dropout_7_layer_call_and_return_conditional_losses_191351

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????(2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????(2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
d
E__inference_dropout_5_layer_call_and_return_conditional_losses_194647

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
d
E__inference_dropout_7_layer_call_and_return_conditional_losses_194701

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
:?????????(2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????(2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????(2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
A__inference_HidB3_layer_call_and_return_conditional_losses_195098

inputs2
matmul_readvariableop_resource:
?F?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?F?*
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
:??????????F: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????F
 
_user_specified_nameinputs
?
]
A__inference_PoolC_layer_call_and_return_conditional_losses_190863

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
?
^
B__inference_PoolC2_layer_call_and_return_conditional_losses_190973

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
E__inference_flatten_8_layer_call_and_return_conditional_losses_191389

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????O  2
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
:?????????":W S
/
_output_shapes
:?????????"
 
_user_specified_nameinputs
?
b
C__inference_dropout_layer_call_and_return_conditional_losses_194166

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
:?????????I?2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????I?*
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
:?????????I?2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????I?2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????I?2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????I?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????I?:W S
/
_output_shapes
:?????????I?
 
_user_specified_nameinputs
?
C
'__inference_PoolA2_layer_call_fn_194256

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
B__inference_PoolA2_layer_call_and_return_conditional_losses_1911352
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
?
c
E__inference_dropout_9_layer_call_and_return_conditional_losses_194743

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????(2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????(2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
a
E__inference_flatten_4_layer_call_and_return_conditional_losses_194896

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????	 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:?????????ȓ2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:?????????ȓ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????H?:W S
/
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?
?
A__inference_ConB3_layer_call_and_return_conditional_losses_191258

inputs8
conv2d_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!(*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!(2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????!(2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????!(2

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
b
F__inference_flatten_10_layer_call_and_return_conditional_losses_194962

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????? 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????D; :W S
/
_output_shapes
:?????????D; 
 
_user_specified_nameinputs
?
d
F__inference_dropout_10_layer_call_and_return_conditional_losses_194770

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????"2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????"2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????":W S
/
_output_shapes
:?????????"
 
_user_specified_nameinputs
?
?
&__inference_HidA2_layer_call_fn_195007

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
A__inference_HidA2_layer_call_and_return_conditional_losses_1914832
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
?
?
%__inference_HidA_layer_call_fn_194987

inputs
unknown:???
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
@__inference_HidA_layer_call_and_return_conditional_losses_1914662
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
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
B
&__inference_PoolB_layer_call_fn_194236

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
:?????????H?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolB_layer_call_and_return_conditional_losses_1910882
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????H?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_4_layer_call_and_return_conditional_losses_192218

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
:?????????D; 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????D; *
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
:?????????D; 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????D; 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????D; 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????D; 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????D; :W S
/
_output_shapes
:?????????D; 
 
_user_specified_nameinputs
?
B
&__inference_PoolA_layer_call_fn_194144

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
A__inference_PoolA_layer_call_and_return_conditional_losses_1907972
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
B__inference_PoolA3_layer_call_and_return_conditional_losses_190885

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
A__inference_ConA4_layer_call_and_return_conditional_losses_194441

inputs8
conv2d_readvariableop_resource:((-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:((*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????(2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
c
*__inference_dropout_2_layer_call_fn_194430

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
:?????????G>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_1923262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????G>2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????G>22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????G>
 
_user_specified_nameinputs
?
?
A__inference_HidB2_layer_call_and_return_conditional_losses_195078

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
?
F
*__inference_dropout_5_layer_call_fn_194652

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
E__inference_dropout_5_layer_call_and_return_conditional_losses_1913652
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
?
^
B__inference_PoolB3_layer_call_and_return_conditional_losses_191303

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????(*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????!(:W S
/
_output_shapes
:?????????!(
 
_user_specified_nameinputs
?
?
A__inference_HidC2_layer_call_and_return_conditional_losses_191602

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
?
]
A__inference_PoolE_layer_call_and_return_conditional_losses_194615

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
E__inference_dropout_7_layer_call_and_return_conditional_losses_194689

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????(2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????(2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
A__inference_HidA4_layer_call_and_return_conditional_losses_195038

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_7_layer_call_fn_194706

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
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_1913512
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
F
*__inference_flatten_6_layer_call_fn_194923

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
:??????????F* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_1914052
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????F2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
F
*__inference_flatten_2_layer_call_fn_194879

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
:??????????F* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_1914372
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????F2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
A__inference_HidC2_layer_call_and_return_conditional_losses_195138

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
??
?<
__inference__traced_save_195778
file_prefix*
&savev2_cona_kernel_read_readvariableop(
$savev2_cona_bias_read_readvariableop*
&savev2_conb_kernel_read_readvariableop(
$savev2_conb_bias_read_readvariableop+
'savev2_cona2_kernel_read_readvariableop)
%savev2_cona2_bias_read_readvariableop*
&savev2_conc_kernel_read_readvariableop(
$savev2_conc_bias_read_readvariableop+
'savev2_cona3_kernel_read_readvariableop)
%savev2_cona3_bias_read_readvariableop+
'savev2_conb2_kernel_read_readvariableop)
%savev2_conb2_bias_read_readvariableop+
'savev2_cona4_kernel_read_readvariableop)
%savev2_cona4_bias_read_readvariableop+
'savev2_conb3_kernel_read_readvariableop)
%savev2_conb3_bias_read_readvariableop+
'savev2_conc2_kernel_read_readvariableop)
%savev2_conc2_bias_read_readvariableop*
&savev2_cond_kernel_read_readvariableop(
$savev2_cond_bias_read_readvariableop*
&savev2_cone_kernel_read_readvariableop(
$savev2_cone_bias_read_readvariableop*
&savev2_hida_kernel_read_readvariableop(
$savev2_hida_bias_read_readvariableop+
'savev2_hida2_kernel_read_readvariableop)
%savev2_hida2_bias_read_readvariableop+
'savev2_hida3_kernel_read_readvariableop)
%savev2_hida3_bias_read_readvariableop+
'savev2_hida4_kernel_read_readvariableop)
%savev2_hida4_bias_read_readvariableop*
&savev2_hidb_kernel_read_readvariableop(
$savev2_hidb_bias_read_readvariableop+
'savev2_hidb2_kernel_read_readvariableop)
%savev2_hidb2_bias_read_readvariableop+
'savev2_hidb3_kernel_read_readvariableop)
%savev2_hidb3_bias_read_readvariableop*
&savev2_hidc_kernel_read_readvariableop(
$savev2_hidc_bias_read_readvariableop+
'savev2_hidc2_kernel_read_readvariableop)
%savev2_hidc2_bias_read_readvariableop+
'savev2_hidd1_kernel_read_readvariableop)
%savev2_hidd1_bias_read_readvariableop+
'savev2_hide1_kernel_read_readvariableop)
%savev2_hide1_bias_read_readvariableop*
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
-savev2_adam_conb_kernel_m_read_readvariableop/
+savev2_adam_conb_bias_m_read_readvariableop2
.savev2_adam_cona2_kernel_m_read_readvariableop0
,savev2_adam_cona2_bias_m_read_readvariableop1
-savev2_adam_conc_kernel_m_read_readvariableop/
+savev2_adam_conc_bias_m_read_readvariableop2
.savev2_adam_cona3_kernel_m_read_readvariableop0
,savev2_adam_cona3_bias_m_read_readvariableop2
.savev2_adam_conb2_kernel_m_read_readvariableop0
,savev2_adam_conb2_bias_m_read_readvariableop2
.savev2_adam_cona4_kernel_m_read_readvariableop0
,savev2_adam_cona4_bias_m_read_readvariableop2
.savev2_adam_conb3_kernel_m_read_readvariableop0
,savev2_adam_conb3_bias_m_read_readvariableop2
.savev2_adam_conc2_kernel_m_read_readvariableop0
,savev2_adam_conc2_bias_m_read_readvariableop1
-savev2_adam_cond_kernel_m_read_readvariableop/
+savev2_adam_cond_bias_m_read_readvariableop1
-savev2_adam_cone_kernel_m_read_readvariableop/
+savev2_adam_cone_bias_m_read_readvariableop1
-savev2_adam_hida_kernel_m_read_readvariableop/
+savev2_adam_hida_bias_m_read_readvariableop2
.savev2_adam_hida2_kernel_m_read_readvariableop0
,savev2_adam_hida2_bias_m_read_readvariableop2
.savev2_adam_hida3_kernel_m_read_readvariableop0
,savev2_adam_hida3_bias_m_read_readvariableop2
.savev2_adam_hida4_kernel_m_read_readvariableop0
,savev2_adam_hida4_bias_m_read_readvariableop1
-savev2_adam_hidb_kernel_m_read_readvariableop/
+savev2_adam_hidb_bias_m_read_readvariableop2
.savev2_adam_hidb2_kernel_m_read_readvariableop0
,savev2_adam_hidb2_bias_m_read_readvariableop2
.savev2_adam_hidb3_kernel_m_read_readvariableop0
,savev2_adam_hidb3_bias_m_read_readvariableop1
-savev2_adam_hidc_kernel_m_read_readvariableop/
+savev2_adam_hidc_bias_m_read_readvariableop2
.savev2_adam_hidc2_kernel_m_read_readvariableop0
,savev2_adam_hidc2_bias_m_read_readvariableop2
.savev2_adam_hidd1_kernel_m_read_readvariableop0
,savev2_adam_hidd1_bias_m_read_readvariableop2
.savev2_adam_hide1_kernel_m_read_readvariableop0
,savev2_adam_hide1_bias_m_read_readvariableop1
-savev2_adam_hid1_kernel_m_read_readvariableop/
+savev2_adam_hid1_bias_m_read_readvariableop1
-savev2_adam_hid2_kernel_m_read_readvariableop/
+savev2_adam_hid2_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop1
-savev2_adam_cona_kernel_v_read_readvariableop/
+savev2_adam_cona_bias_v_read_readvariableop1
-savev2_adam_conb_kernel_v_read_readvariableop/
+savev2_adam_conb_bias_v_read_readvariableop2
.savev2_adam_cona2_kernel_v_read_readvariableop0
,savev2_adam_cona2_bias_v_read_readvariableop1
-savev2_adam_conc_kernel_v_read_readvariableop/
+savev2_adam_conc_bias_v_read_readvariableop2
.savev2_adam_cona3_kernel_v_read_readvariableop0
,savev2_adam_cona3_bias_v_read_readvariableop2
.savev2_adam_conb2_kernel_v_read_readvariableop0
,savev2_adam_conb2_bias_v_read_readvariableop2
.savev2_adam_cona4_kernel_v_read_readvariableop0
,savev2_adam_cona4_bias_v_read_readvariableop2
.savev2_adam_conb3_kernel_v_read_readvariableop0
,savev2_adam_conb3_bias_v_read_readvariableop2
.savev2_adam_conc2_kernel_v_read_readvariableop0
,savev2_adam_conc2_bias_v_read_readvariableop1
-savev2_adam_cond_kernel_v_read_readvariableop/
+savev2_adam_cond_bias_v_read_readvariableop1
-savev2_adam_cone_kernel_v_read_readvariableop/
+savev2_adam_cone_bias_v_read_readvariableop1
-savev2_adam_hida_kernel_v_read_readvariableop/
+savev2_adam_hida_bias_v_read_readvariableop2
.savev2_adam_hida2_kernel_v_read_readvariableop0
,savev2_adam_hida2_bias_v_read_readvariableop2
.savev2_adam_hida3_kernel_v_read_readvariableop0
,savev2_adam_hida3_bias_v_read_readvariableop2
.savev2_adam_hida4_kernel_v_read_readvariableop0
,savev2_adam_hida4_bias_v_read_readvariableop1
-savev2_adam_hidb_kernel_v_read_readvariableop/
+savev2_adam_hidb_bias_v_read_readvariableop2
.savev2_adam_hidb2_kernel_v_read_readvariableop0
,savev2_adam_hidb2_bias_v_read_readvariableop2
.savev2_adam_hidb3_kernel_v_read_readvariableop0
,savev2_adam_hidb3_bias_v_read_readvariableop1
-savev2_adam_hidc_kernel_v_read_readvariableop/
+savev2_adam_hidc_bias_v_read_readvariableop2
.savev2_adam_hidc2_kernel_v_read_readvariableop0
,savev2_adam_hidc2_bias_v_read_readvariableop2
.savev2_adam_hidd1_kernel_v_read_readvariableop0
,savev2_adam_hidd1_bias_v_read_readvariableop2
.savev2_adam_hide1_kernel_v_read_readvariableop0
,savev2_adam_hide1_bias_v_read_readvariableop1
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
ShardedFilename?[
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?Z
value?ZB?Z?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?9
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_cona_kernel_read_readvariableop$savev2_cona_bias_read_readvariableop&savev2_conb_kernel_read_readvariableop$savev2_conb_bias_read_readvariableop'savev2_cona2_kernel_read_readvariableop%savev2_cona2_bias_read_readvariableop&savev2_conc_kernel_read_readvariableop$savev2_conc_bias_read_readvariableop'savev2_cona3_kernel_read_readvariableop%savev2_cona3_bias_read_readvariableop'savev2_conb2_kernel_read_readvariableop%savev2_conb2_bias_read_readvariableop'savev2_cona4_kernel_read_readvariableop%savev2_cona4_bias_read_readvariableop'savev2_conb3_kernel_read_readvariableop%savev2_conb3_bias_read_readvariableop'savev2_conc2_kernel_read_readvariableop%savev2_conc2_bias_read_readvariableop&savev2_cond_kernel_read_readvariableop$savev2_cond_bias_read_readvariableop&savev2_cone_kernel_read_readvariableop$savev2_cone_bias_read_readvariableop&savev2_hida_kernel_read_readvariableop$savev2_hida_bias_read_readvariableop'savev2_hida2_kernel_read_readvariableop%savev2_hida2_bias_read_readvariableop'savev2_hida3_kernel_read_readvariableop%savev2_hida3_bias_read_readvariableop'savev2_hida4_kernel_read_readvariableop%savev2_hida4_bias_read_readvariableop&savev2_hidb_kernel_read_readvariableop$savev2_hidb_bias_read_readvariableop'savev2_hidb2_kernel_read_readvariableop%savev2_hidb2_bias_read_readvariableop'savev2_hidb3_kernel_read_readvariableop%savev2_hidb3_bias_read_readvariableop&savev2_hidc_kernel_read_readvariableop$savev2_hidc_bias_read_readvariableop'savev2_hidc2_kernel_read_readvariableop%savev2_hidc2_bias_read_readvariableop'savev2_hidd1_kernel_read_readvariableop%savev2_hidd1_bias_read_readvariableop'savev2_hide1_kernel_read_readvariableop%savev2_hide1_bias_read_readvariableop&savev2_hid1_kernel_read_readvariableop$savev2_hid1_bias_read_readvariableop&savev2_hid2_kernel_read_readvariableop$savev2_hid2_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop-savev2_adam_cona_kernel_m_read_readvariableop+savev2_adam_cona_bias_m_read_readvariableop-savev2_adam_conb_kernel_m_read_readvariableop+savev2_adam_conb_bias_m_read_readvariableop.savev2_adam_cona2_kernel_m_read_readvariableop,savev2_adam_cona2_bias_m_read_readvariableop-savev2_adam_conc_kernel_m_read_readvariableop+savev2_adam_conc_bias_m_read_readvariableop.savev2_adam_cona3_kernel_m_read_readvariableop,savev2_adam_cona3_bias_m_read_readvariableop.savev2_adam_conb2_kernel_m_read_readvariableop,savev2_adam_conb2_bias_m_read_readvariableop.savev2_adam_cona4_kernel_m_read_readvariableop,savev2_adam_cona4_bias_m_read_readvariableop.savev2_adam_conb3_kernel_m_read_readvariableop,savev2_adam_conb3_bias_m_read_readvariableop.savev2_adam_conc2_kernel_m_read_readvariableop,savev2_adam_conc2_bias_m_read_readvariableop-savev2_adam_cond_kernel_m_read_readvariableop+savev2_adam_cond_bias_m_read_readvariableop-savev2_adam_cone_kernel_m_read_readvariableop+savev2_adam_cone_bias_m_read_readvariableop-savev2_adam_hida_kernel_m_read_readvariableop+savev2_adam_hida_bias_m_read_readvariableop.savev2_adam_hida2_kernel_m_read_readvariableop,savev2_adam_hida2_bias_m_read_readvariableop.savev2_adam_hida3_kernel_m_read_readvariableop,savev2_adam_hida3_bias_m_read_readvariableop.savev2_adam_hida4_kernel_m_read_readvariableop,savev2_adam_hida4_bias_m_read_readvariableop-savev2_adam_hidb_kernel_m_read_readvariableop+savev2_adam_hidb_bias_m_read_readvariableop.savev2_adam_hidb2_kernel_m_read_readvariableop,savev2_adam_hidb2_bias_m_read_readvariableop.savev2_adam_hidb3_kernel_m_read_readvariableop,savev2_adam_hidb3_bias_m_read_readvariableop-savev2_adam_hidc_kernel_m_read_readvariableop+savev2_adam_hidc_bias_m_read_readvariableop.savev2_adam_hidc2_kernel_m_read_readvariableop,savev2_adam_hidc2_bias_m_read_readvariableop.savev2_adam_hidd1_kernel_m_read_readvariableop,savev2_adam_hidd1_bias_m_read_readvariableop.savev2_adam_hide1_kernel_m_read_readvariableop,savev2_adam_hide1_bias_m_read_readvariableop-savev2_adam_hid1_kernel_m_read_readvariableop+savev2_adam_hid1_bias_m_read_readvariableop-savev2_adam_hid2_kernel_m_read_readvariableop+savev2_adam_hid2_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop-savev2_adam_cona_kernel_v_read_readvariableop+savev2_adam_cona_bias_v_read_readvariableop-savev2_adam_conb_kernel_v_read_readvariableop+savev2_adam_conb_bias_v_read_readvariableop.savev2_adam_cona2_kernel_v_read_readvariableop,savev2_adam_cona2_bias_v_read_readvariableop-savev2_adam_conc_kernel_v_read_readvariableop+savev2_adam_conc_bias_v_read_readvariableop.savev2_adam_cona3_kernel_v_read_readvariableop,savev2_adam_cona3_bias_v_read_readvariableop.savev2_adam_conb2_kernel_v_read_readvariableop,savev2_adam_conb2_bias_v_read_readvariableop.savev2_adam_cona4_kernel_v_read_readvariableop,savev2_adam_cona4_bias_v_read_readvariableop.savev2_adam_conb3_kernel_v_read_readvariableop,savev2_adam_conb3_bias_v_read_readvariableop.savev2_adam_conc2_kernel_v_read_readvariableop,savev2_adam_conc2_bias_v_read_readvariableop-savev2_adam_cond_kernel_v_read_readvariableop+savev2_adam_cond_bias_v_read_readvariableop-savev2_adam_cone_kernel_v_read_readvariableop+savev2_adam_cone_bias_v_read_readvariableop-savev2_adam_hida_kernel_v_read_readvariableop+savev2_adam_hida_bias_v_read_readvariableop.savev2_adam_hida2_kernel_v_read_readvariableop,savev2_adam_hida2_bias_v_read_readvariableop.savev2_adam_hida3_kernel_v_read_readvariableop,savev2_adam_hida3_bias_v_read_readvariableop.savev2_adam_hida4_kernel_v_read_readvariableop,savev2_adam_hida4_bias_v_read_readvariableop-savev2_adam_hidb_kernel_v_read_readvariableop+savev2_adam_hidb_bias_v_read_readvariableop.savev2_adam_hidb2_kernel_v_read_readvariableop,savev2_adam_hidb2_bias_v_read_readvariableop.savev2_adam_hidb3_kernel_v_read_readvariableop,savev2_adam_hidb3_bias_v_read_readvariableop-savev2_adam_hidc_kernel_v_read_readvariableop+savev2_adam_hidc_bias_v_read_readvariableop.savev2_adam_hidc2_kernel_v_read_readvariableop,savev2_adam_hidc2_bias_v_read_readvariableop.savev2_adam_hidd1_kernel_v_read_readvariableop,savev2_adam_hidd1_bias_v_read_readvariableop.savev2_adam_hide1_kernel_v_read_readvariableop,savev2_adam_hide1_bias_v_read_readvariableop-savev2_adam_hid1_kernel_v_read_readvariableop+savev2_adam_hid1_bias_v_read_readvariableop-savev2_adam_hid2_kernel_v_read_readvariableop+savev2_adam_hid2_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes?
?2?	2
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
?: :::::::	::(:(:::((:(:(:(:::	:: : :???:?:???:?:
?F?:?:
??:?:ȓ?:?:???:?:
?F?:?:???:?:???:?:???:?:???:?:
??:?:
??:?:
??:?: : : : : : : : : :::::::	::(:(:::((:(:(:(:::	:: : :???:?:???:?:
?F?:?:
??:?:ȓ?:?:???:?:
?F?:?:???:?:???:?:???:?:???:?:
??:?:
??:?:
??:?:::::::	::(:(:::((:(:(:(:::	:: : :???:?:???:?:
?F?:?:
??:?:ȓ?:?:???:?:
?F?:?:???:?:???:?:???:?:???:?:
??:?:
??:?:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:	: 

_output_shapes
::,	(
&
_output_shapes
:(: 


_output_shapes
:(:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:((: 

_output_shapes
:(:,(
&
_output_shapes
:(: 

_output_shapes
:(:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:	: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :'#
!
_output_shapes
:???:!

_output_shapes	
:?:'#
!
_output_shapes
:???:!

_output_shapes	
:?:&"
 
_output_shapes
:
?F?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:'#
!
_output_shapes
:ȓ?:! 

_output_shapes	
:?:'!#
!
_output_shapes
:???:!"

_output_shapes	
:?:&#"
 
_output_shapes
:
?F?:!$

_output_shapes	
:?:'%#
!
_output_shapes
:???:!&

_output_shapes	
:?:''#
!
_output_shapes
:???:!(

_output_shapes	
:?:')#
!
_output_shapes
:???:!*

_output_shapes	
:?:'+#
!
_output_shapes
:???:!,

_output_shapes	
:?:&-"
 
_output_shapes
:
??:!.

_output_shapes	
:?:&/"
 
_output_shapes
:
??:!0

_output_shapes	
:?:&1"
 
_output_shapes
:
??:!2

_output_shapes	
:?:3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7
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
: :,<(
&
_output_shapes
:: =

_output_shapes
::,>(
&
_output_shapes
:: ?

_output_shapes
::,@(
&
_output_shapes
:: A

_output_shapes
::,B(
&
_output_shapes
:	: C

_output_shapes
::,D(
&
_output_shapes
:(: E

_output_shapes
:(:,F(
&
_output_shapes
:: G

_output_shapes
::,H(
&
_output_shapes
:((: I

_output_shapes
:(:,J(
&
_output_shapes
:(: K

_output_shapes
:(:,L(
&
_output_shapes
:: M

_output_shapes
::,N(
&
_output_shapes
:	: O

_output_shapes
::,P(
&
_output_shapes
: : Q

_output_shapes
: :'R#
!
_output_shapes
:???:!S

_output_shapes	
:?:'T#
!
_output_shapes
:???:!U

_output_shapes	
:?:&V"
 
_output_shapes
:
?F?:!W

_output_shapes	
:?:&X"
 
_output_shapes
:
??:!Y

_output_shapes	
:?:'Z#
!
_output_shapes
:ȓ?:![

_output_shapes	
:?:'\#
!
_output_shapes
:???:!]

_output_shapes	
:?:&^"
 
_output_shapes
:
?F?:!_

_output_shapes	
:?:'`#
!
_output_shapes
:???:!a

_output_shapes	
:?:'b#
!
_output_shapes
:???:!c

_output_shapes	
:?:'d#
!
_output_shapes
:???:!e

_output_shapes	
:?:'f#
!
_output_shapes
:???:!g

_output_shapes	
:?:&h"
 
_output_shapes
:
??:!i

_output_shapes	
:?:&j"
 
_output_shapes
:
??:!k

_output_shapes	
:?:&l"
 
_output_shapes
:
??:!m

_output_shapes	
:?:,n(
&
_output_shapes
:: o

_output_shapes
::,p(
&
_output_shapes
:: q

_output_shapes
::,r(
&
_output_shapes
:: s

_output_shapes
::,t(
&
_output_shapes
:	: u

_output_shapes
::,v(
&
_output_shapes
:(: w

_output_shapes
:(:,x(
&
_output_shapes
:: y

_output_shapes
::,z(
&
_output_shapes
:((: {

_output_shapes
:(:,|(
&
_output_shapes
:(: }

_output_shapes
:(:,~(
&
_output_shapes
:: 

_output_shapes
::-?(
&
_output_shapes
:	:!?

_output_shapes
::-?(
&
_output_shapes
: :!?

_output_shapes
: :(?#
!
_output_shapes
:???:"?
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
?F?:"?

_output_shapes	
:?:'?"
 
_output_shapes
:
??:"?

_output_shapes	
:?:(?#
!
_output_shapes
:ȓ?:"?
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
?F?:"?

_output_shapes	
:?:(?#
!
_output_shapes
:???:"?

_output_shapes	
:?:(?#
!
_output_shapes
:???:"?

_output_shapes	
:?:(?#
!
_output_shapes
:???:"?

_output_shapes	
:?:(?#
!
_output_shapes
:???:"?

_output_shapes	
:?:'?"
 
_output_shapes
:
??:"?

_output_shapes	
:?:'?"
 
_output_shapes
:
??:"?

_output_shapes	
:?:'?"
 
_output_shapes
:
??:"?

_output_shapes	
:?:?

_output_shapes
: 
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_191453

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"??????  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????I?:W S
/
_output_shapes
:?????????I?
 
_user_specified_nameinputs
?
F
*__inference_dropout_6_layer_call_fn_194679

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
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_1913582
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
@__inference_ConB_layer_call_and_return_conditional_losses_194187

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????2

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
?
?
&__inference_ConA3_layer_call_fn_194323

inputs!
unknown:(
	unknown_0:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConA3_layer_call_and_return_conditional_losses_1911712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????!(2

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
?
B
&__inference_PoolC_layer_call_fn_194358

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
A__inference_PoolC_layer_call_and_return_conditional_losses_1908632
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
@__inference_Hid1_layer_call_and_return_conditional_losses_195229

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
׽
?'
!__inference__wrapped_model_190788	
inputC
)model_cona_conv2d_readvariableop_resource:8
*model_cona_biasadd_readvariableop_resource:C
)model_conb_conv2d_readvariableop_resource:8
*model_conb_biasadd_readvariableop_resource:D
*model_cona2_conv2d_readvariableop_resource:9
+model_cona2_biasadd_readvariableop_resource:C
)model_conc_conv2d_readvariableop_resource:	8
*model_conc_biasadd_readvariableop_resource:D
*model_conb2_conv2d_readvariableop_resource:9
+model_conb2_biasadd_readvariableop_resource:D
*model_cona3_conv2d_readvariableop_resource:(9
+model_cona3_biasadd_readvariableop_resource:(C
)model_cone_conv2d_readvariableop_resource: 8
*model_cone_biasadd_readvariableop_resource: C
)model_cond_conv2d_readvariableop_resource:	8
*model_cond_biasadd_readvariableop_resource:D
*model_conc2_conv2d_readvariableop_resource:9
+model_conc2_biasadd_readvariableop_resource:D
*model_conb3_conv2d_readvariableop_resource:(9
+model_conb3_biasadd_readvariableop_resource:(D
*model_cona4_conv2d_readvariableop_resource:((9
+model_cona4_biasadd_readvariableop_resource:(>
)model_hida_matmul_readvariableop_resource:???9
*model_hida_biasadd_readvariableop_resource:	??
*model_hida2_matmul_readvariableop_resource:???:
+model_hida2_biasadd_readvariableop_resource:	?>
*model_hida3_matmul_readvariableop_resource:
?F?:
+model_hida3_biasadd_readvariableop_resource:	?>
*model_hida4_matmul_readvariableop_resource:
??:
+model_hida4_biasadd_readvariableop_resource:	?>
)model_hidb_matmul_readvariableop_resource:ȓ?9
*model_hidb_biasadd_readvariableop_resource:	??
*model_hidb2_matmul_readvariableop_resource:???:
+model_hidb2_biasadd_readvariableop_resource:	?>
*model_hidb3_matmul_readvariableop_resource:
?F?:
+model_hidb3_biasadd_readvariableop_resource:	?>
)model_hidc_matmul_readvariableop_resource:???9
*model_hidc_biasadd_readvariableop_resource:	??
*model_hidc2_matmul_readvariableop_resource:???:
+model_hidc2_biasadd_readvariableop_resource:	??
*model_hidd1_matmul_readvariableop_resource:???:
+model_hidd1_biasadd_readvariableop_resource:	??
*model_hide1_matmul_readvariableop_resource:???:
+model_hide1_biasadd_readvariableop_resource:	?=
)model_hid1_matmul_readvariableop_resource:
??9
*model_hid1_biasadd_readvariableop_resource:	?=
)model_hid2_matmul_readvariableop_resource:
??9
*model_hid2_biasadd_readvariableop_resource:	??
+model_output_matmul_readvariableop_resource:
??;
,model_output_biasadd_readvariableop_resource:	?
identity??!model/ConA/BiasAdd/ReadVariableOp? model/ConA/Conv2D/ReadVariableOp?"model/ConA2/BiasAdd/ReadVariableOp?!model/ConA2/Conv2D/ReadVariableOp?"model/ConA3/BiasAdd/ReadVariableOp?!model/ConA3/Conv2D/ReadVariableOp?"model/ConA4/BiasAdd/ReadVariableOp?!model/ConA4/Conv2D/ReadVariableOp?!model/ConB/BiasAdd/ReadVariableOp? model/ConB/Conv2D/ReadVariableOp?"model/ConB2/BiasAdd/ReadVariableOp?!model/ConB2/Conv2D/ReadVariableOp?"model/ConB3/BiasAdd/ReadVariableOp?!model/ConB3/Conv2D/ReadVariableOp?!model/ConC/BiasAdd/ReadVariableOp? model/ConC/Conv2D/ReadVariableOp?"model/ConC2/BiasAdd/ReadVariableOp?!model/ConC2/Conv2D/ReadVariableOp?!model/ConD/BiasAdd/ReadVariableOp? model/ConD/Conv2D/ReadVariableOp?!model/ConE/BiasAdd/ReadVariableOp? model/ConE/Conv2D/ReadVariableOp?!model/Hid1/BiasAdd/ReadVariableOp? model/Hid1/MatMul/ReadVariableOp?!model/Hid2/BiasAdd/ReadVariableOp? model/Hid2/MatMul/ReadVariableOp?!model/HidA/BiasAdd/ReadVariableOp? model/HidA/MatMul/ReadVariableOp?"model/HidA2/BiasAdd/ReadVariableOp?!model/HidA2/MatMul/ReadVariableOp?"model/HidA3/BiasAdd/ReadVariableOp?!model/HidA3/MatMul/ReadVariableOp?"model/HidA4/BiasAdd/ReadVariableOp?!model/HidA4/MatMul/ReadVariableOp?!model/HidB/BiasAdd/ReadVariableOp? model/HidB/MatMul/ReadVariableOp?"model/HidB2/BiasAdd/ReadVariableOp?!model/HidB2/MatMul/ReadVariableOp?"model/HidB3/BiasAdd/ReadVariableOp?!model/HidB3/MatMul/ReadVariableOp?!model/HidC/BiasAdd/ReadVariableOp? model/HidC/MatMul/ReadVariableOp?"model/HidC2/BiasAdd/ReadVariableOp?!model/HidC2/MatMul/ReadVariableOp?"model/HidD1/BiasAdd/ReadVariableOp?!model/HidD1/MatMul/ReadVariableOp?"model/HidE1/BiasAdd/ReadVariableOp?!model/HidE1/MatMul/ReadVariableOp?#model/output/BiasAdd/ReadVariableOp?"model/output/MatMul/ReadVariableOp?
 model/ConA/Conv2D/ReadVariableOpReadVariableOp)model_cona_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 model/ConA/Conv2D/ReadVariableOp?
model/ConA/Conv2DConv2Dinput(model/ConA/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
model/ConA/Conv2D?
!model/ConA/BiasAdd/ReadVariableOpReadVariableOp*model_cona_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!model/ConA/BiasAdd/ReadVariableOp?
model/ConA/BiasAddBiasAddmodel/ConA/Conv2D:output:0)model/ConA/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
model/ConA/BiasAdd?
model/ConA/ReluRelumodel/ConA/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
model/ConA/Relu?
model/PoolA/MaxPoolMaxPoolmodel/ConA/Relu:activations:0*/
_output_shapes
:?????????I?*
ksize
*
paddingVALID*
strides
2
model/PoolA/MaxPool?
 model/ConB/Conv2D/ReadVariableOpReadVariableOp)model_conb_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 model/ConB/Conv2D/ReadVariableOp?
model/ConB/Conv2DConv2Dinput(model/ConB/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
model/ConB/Conv2D?
!model/ConB/BiasAdd/ReadVariableOpReadVariableOp*model_conb_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!model/ConB/BiasAdd/ReadVariableOp?
model/ConB/BiasAddBiasAddmodel/ConB/Conv2D:output:0)model/ConB/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
model/ConB/BiasAdd?
model/ConB/ReluRelumodel/ConB/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
model/ConB/Relu?
model/dropout/IdentityIdentitymodel/PoolA/MaxPool:output:0*
T0*/
_output_shapes
:?????????I?2
model/dropout/Identity?
model/PoolB/MaxPoolMaxPoolmodel/ConB/Relu:activations:0*/
_output_shapes
:?????????H?*
ksize
*
paddingVALID*
strides
2
model/PoolB/MaxPool?
!model/ConA2/Conv2D/ReadVariableOpReadVariableOp*model_cona2_conv2d_readvariableop_resource*&
_output_shapes
:*
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
 model/ConC/Conv2D/ReadVariableOpReadVariableOp)model_conc_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02"
 model/ConC/Conv2D/ReadVariableOp?
model/ConC/Conv2DConv2Dinput(model/ConC/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
model/ConC/Conv2D?
!model/ConC/BiasAdd/ReadVariableOpReadVariableOp*model_conc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!model/ConC/BiasAdd/ReadVariableOp?
model/ConC/BiasAddBiasAddmodel/ConC/Conv2D:output:0)model/ConC/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
model/ConC/BiasAdd?
model/ConC/ReluRelumodel/ConC/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
model/ConC/Relu?
model/dropout_1/IdentityIdentitymodel/PoolB/MaxPool:output:0*
T0*/
_output_shapes
:?????????H?2
model/dropout_1/Identity?
model/PoolA2/MaxPoolMaxPoolmodel/ConA2/Relu:activations:0*/
_output_shapes
:?????????#*
ksize
*
paddingVALID*
strides
2
model/PoolA2/MaxPool?
model/PoolC/MaxPoolMaxPoolmodel/ConC/Relu:activations:0*/
_output_shapes
:?????????G>*
ksize
*
paddingVALID*
strides
2
model/PoolC/MaxPool?
!model/ConB2/Conv2D/ReadVariableOpReadVariableOp*model_conb2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02#
!model/ConB2/Conv2D/ReadVariableOp?
model/ConB2/Conv2DConv2D!model/dropout_1/Identity:output:0)model/ConB2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????F=*
paddingVALID*
strides
2
model/ConB2/Conv2D?
"model/ConB2/BiasAdd/ReadVariableOpReadVariableOp+model_conb2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/ConB2/BiasAdd/ReadVariableOp?
model/ConB2/BiasAddBiasAddmodel/ConB2/Conv2D:output:0*model/ConB2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????F=2
model/ConB2/BiasAdd?
model/ConB2/ReluRelumodel/ConB2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????F=2
model/ConB2/Relu?
!model/ConA3/Conv2D/ReadVariableOpReadVariableOp*model_cona3_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02#
!model/ConA3/Conv2D/ReadVariableOp?
model/ConA3/Conv2DConv2Dmodel/PoolA2/MaxPool:output:0)model/ConA3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!(*
paddingVALID*
strides
2
model/ConA3/Conv2D?
"model/ConA3/BiasAdd/ReadVariableOpReadVariableOp+model_cona3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02$
"model/ConA3/BiasAdd/ReadVariableOp?
model/ConA3/BiasAddBiasAddmodel/ConA3/Conv2D:output:0*model/ConA3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!(2
model/ConA3/BiasAdd?
model/ConA3/ReluRelumodel/ConA3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????!(2
model/ConA3/Relu?
model/dropout_2/IdentityIdentitymodel/PoolC/MaxPool:output:0*
T0*/
_output_shapes
:?????????G>2
model/dropout_2/Identity?
model/PoolB2/MaxPoolMaxPoolmodel/ConB2/Relu:activations:0*/
_output_shapes
:?????????#*
ksize
*
paddingVALID*
strides
2
model/PoolB2/MaxPool?
model/PoolA3/MaxPoolMaxPoolmodel/ConA3/Relu:activations:0*/
_output_shapes
:?????????(*
ksize
*
paddingVALID*
strides
2
model/PoolA3/MaxPool?
 model/ConE/Conv2D/ReadVariableOpReadVariableOp)model_cone_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02"
 model/ConE/Conv2D/ReadVariableOp?
model/ConE/Conv2DConv2Dinput(model/ConE/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
2
model/ConE/Conv2D?
!model/ConE/BiasAdd/ReadVariableOpReadVariableOp*model_cone_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!model/ConE/BiasAdd/ReadVariableOp?
model/ConE/BiasAddBiasAddmodel/ConE/Conv2D:output:0)model/ConE/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
model/ConE/BiasAdd?
model/ConE/ReluRelumodel/ConE/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
model/ConE/Relu?
 model/ConD/Conv2D/ReadVariableOpReadVariableOp)model_cond_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02"
 model/ConD/Conv2D/ReadVariableOp?
model/ConD/Conv2DConv2Dinput(model/ConD/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
model/ConD/Conv2D?
!model/ConD/BiasAdd/ReadVariableOpReadVariableOp*model_cond_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!model/ConD/BiasAdd/ReadVariableOp?
model/ConD/BiasAddBiasAddmodel/ConD/Conv2D:output:0)model/ConD/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
model/ConD/BiasAdd?
model/ConD/ReluRelumodel/ConD/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
model/ConD/Relu?
!model/ConC2/Conv2D/ReadVariableOpReadVariableOp*model_conc2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02#
!model/ConC2/Conv2D/ReadVariableOp?
model/ConC2/Conv2DConv2D!model/dropout_2/Identity:output:0)model/ConC2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????E<*
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
:?????????E<2
model/ConC2/BiasAdd?
model/ConC2/ReluRelumodel/ConC2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????E<2
model/ConC2/Relu?
!model/ConB3/Conv2D/ReadVariableOpReadVariableOp*model_conb3_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02#
!model/ConB3/Conv2D/ReadVariableOp?
model/ConB3/Conv2DConv2Dmodel/PoolB2/MaxPool:output:0)model/ConB3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!(*
paddingVALID*
strides
2
model/ConB3/Conv2D?
"model/ConB3/BiasAdd/ReadVariableOpReadVariableOp+model_conb3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02$
"model/ConB3/BiasAdd/ReadVariableOp?
model/ConB3/BiasAddBiasAddmodel/ConB3/Conv2D:output:0*model/ConB3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!(2
model/ConB3/BiasAdd?
model/ConB3/ReluRelumodel/ConB3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????!(2
model/ConB3/Relu?
!model/ConA4/Conv2D/ReadVariableOpReadVariableOp*model_cona4_conv2d_readvariableop_resource*&
_output_shapes
:((*
dtype02#
!model/ConA4/Conv2D/ReadVariableOp?
model/ConA4/Conv2DConv2Dmodel/PoolA3/MaxPool:output:0)model/ConA4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingVALID*
strides
2
model/ConA4/Conv2D?
"model/ConA4/BiasAdd/ReadVariableOpReadVariableOp+model_cona4_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02$
"model/ConA4/BiasAdd/ReadVariableOp?
model/ConA4/BiasAddBiasAddmodel/ConA4/Conv2D:output:0*model/ConA4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2
model/ConA4/BiasAdd?
model/ConA4/ReluRelumodel/ConA4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(2
model/ConA4/Relu?
model/PoolE/MaxPoolMaxPoolmodel/ConE/Relu:activations:0*/
_output_shapes
:?????????D; *
ksize
*
paddingVALID*
strides
2
model/PoolE/MaxPool?
model/PoolD/MaxPoolMaxPoolmodel/ConD/Relu:activations:0*/
_output_shapes
:?????????G=*
ksize
*
paddingVALID*
strides
2
model/PoolD/MaxPool?
model/PoolC2/MaxPoolMaxPoolmodel/ConC2/Relu:activations:0*/
_output_shapes
:?????????"*
ksize
*
paddingVALID*
strides
2
model/PoolC2/MaxPool?
model/PoolB3/MaxPoolMaxPoolmodel/ConB3/Relu:activations:0*/
_output_shapes
:?????????(*
ksize
*
paddingVALID*
strides
2
model/PoolB3/MaxPool?
model/PoolA4/MaxPoolMaxPoolmodel/ConA4/Relu:activations:0*/
_output_shapes
:?????????(*
ksize
*
paddingVALID*
strides
2
model/PoolA4/MaxPool?
model/dropout_4/IdentityIdentitymodel/PoolE/MaxPool:output:0*
T0*/
_output_shapes
:?????????D; 2
model/dropout_4/Identity?
model/dropout_3/IdentityIdentitymodel/PoolD/MaxPool:output:0*
T0*/
_output_shapes
:?????????G=2
model/dropout_3/Identity?
model/dropout_10/IdentityIdentitymodel/PoolC2/MaxPool:output:0*
T0*/
_output_shapes
:?????????"2
model/dropout_10/Identity?
model/dropout_9/IdentityIdentitymodel/PoolB3/MaxPool:output:0*
T0*/
_output_shapes
:?????????(2
model/dropout_9/Identity?
model/dropout_8/IdentityIdentitymodel/PoolB2/MaxPool:output:0*
T0*/
_output_shapes
:?????????#2
model/dropout_8/Identity?
model/dropout_7/IdentityIdentitymodel/PoolA4/MaxPool:output:0*
T0*/
_output_shapes
:?????????(2
model/dropout_7/Identity?
model/dropout_6/IdentityIdentitymodel/PoolA3/MaxPool:output:0*
T0*/
_output_shapes
:?????????(2
model/dropout_6/Identity?
model/dropout_5/IdentityIdentitymodel/PoolA2/MaxPool:output:0*
T0*/
_output_shapes
:?????????#2
model/dropout_5/Identity?
model/flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????? 2
model/flatten_10/Const?
model/flatten_10/ReshapeReshape!model/dropout_4/Identity:output:0model/flatten_10/Const:output:0*
T0*)
_output_shapes
:???????????2
model/flatten_10/Reshape
model/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????? 2
model/flatten_9/Const?
model/flatten_9/ReshapeReshape!model/dropout_3/Identity:output:0model/flatten_9/Const:output:0*
T0*)
_output_shapes
:???????????2
model/flatten_9/Reshape
model/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????O  2
model/flatten_8/Const?
model/flatten_8/ReshapeReshape"model/dropout_10/Identity:output:0model/flatten_8/Const:output:0*
T0*)
_output_shapes
:???????????2
model/flatten_8/Reshape
model/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????W 2
model/flatten_7/Const?
model/flatten_7/ReshapeReshape!model/dropout_2/Identity:output:0model/flatten_7/Const:output:0*
T0*)
_output_shapes
:???????????2
model/flatten_7/Reshape
model/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? #  2
model/flatten_6/Const?
model/flatten_6/ReshapeReshape!model/dropout_9/Identity:output:0model/flatten_6/Const:output:0*
T0*(
_output_shapes
:??????????F2
model/flatten_6/Reshape
model/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????R  2
model/flatten_5/Const?
model/flatten_5/ReshapeReshape!model/dropout_8/Identity:output:0model/flatten_5/Const:output:0*
T0*)
_output_shapes
:???????????2
model/flatten_5/Reshape
model/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????	 2
model/flatten_4/Const?
model/flatten_4/ReshapeReshapemodel/PoolB/MaxPool:output:0model/flatten_4/Const:output:0*
T0*)
_output_shapes
:?????????ȓ2
model/flatten_4/Reshape
model/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
model/flatten_3/Const?
model/flatten_3/ReshapeReshape!model/dropout_7/Identity:output:0model/flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????2
model/flatten_3/Reshape
model/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? #  2
model/flatten_2/Const?
model/flatten_2/ReshapeReshape!model/dropout_6/Identity:output:0model/flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????F2
model/flatten_2/Reshape
model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????R  2
model/flatten_1/Const?
model/flatten_1/ReshapeReshape!model/dropout_5/Identity:output:0model/flatten_1/Const:output:0*
T0*)
_output_shapes
:???????????2
model/flatten_1/Reshape{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"??????  2
model/flatten/Const?
model/flatten/ReshapeReshapemodel/PoolA/MaxPool:output:0model/flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
model/flatten/Reshape?
 model/HidA/MatMul/ReadVariableOpReadVariableOp)model_hida_matmul_readvariableop_resource*!
_output_shapes
:???*
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
?F?*
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
??*
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
 model/HidB/MatMul/ReadVariableOpReadVariableOp)model_hidb_matmul_readvariableop_resource*!
_output_shapes
:ȓ?*
dtype02"
 model/HidB/MatMul/ReadVariableOp?
model/HidB/MatMulMatMul model/flatten_4/Reshape:output:0(model/HidB/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidB/MatMul?
!model/HidB/BiasAdd/ReadVariableOpReadVariableOp*model_hidb_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!model/HidB/BiasAdd/ReadVariableOp?
model/HidB/BiasAddBiasAddmodel/HidB/MatMul:product:0)model/HidB/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidB/BiasAddz
model/HidB/ReluRelumodel/HidB/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidB/Relu?
!model/HidB2/MatMul/ReadVariableOpReadVariableOp*model_hidb2_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02#
!model/HidB2/MatMul/ReadVariableOp?
model/HidB2/MatMulMatMul model/flatten_5/Reshape:output:0)model/HidB2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidB2/MatMul?
"model/HidB2/BiasAdd/ReadVariableOpReadVariableOp+model_hidb2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"model/HidB2/BiasAdd/ReadVariableOp?
model/HidB2/BiasAddBiasAddmodel/HidB2/MatMul:product:0*model/HidB2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidB2/BiasAdd}
model/HidB2/ReluRelumodel/HidB2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidB2/Relu?
!model/HidB3/MatMul/ReadVariableOpReadVariableOp*model_hidb3_matmul_readvariableop_resource* 
_output_shapes
:
?F?*
dtype02#
!model/HidB3/MatMul/ReadVariableOp?
model/HidB3/MatMulMatMul model/flatten_6/Reshape:output:0)model/HidB3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidB3/MatMul?
"model/HidB3/BiasAdd/ReadVariableOpReadVariableOp+model_hidb3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"model/HidB3/BiasAdd/ReadVariableOp?
model/HidB3/BiasAddBiasAddmodel/HidB3/MatMul:product:0*model/HidB3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidB3/BiasAdd}
model/HidB3/ReluRelumodel/HidB3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidB3/Relu?
 model/HidC/MatMul/ReadVariableOpReadVariableOp)model_hidc_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02"
 model/HidC/MatMul/ReadVariableOp?
model/HidC/MatMulMatMul model/flatten_7/Reshape:output:0(model/HidC/MatMul/ReadVariableOp:value:0*
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
model/HidC2/MatMulMatMul model/flatten_8/Reshape:output:0)model/HidC2/MatMul/ReadVariableOp:value:0*
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
!model/HidD1/MatMul/ReadVariableOpReadVariableOp*model_hidd1_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02#
!model/HidD1/MatMul/ReadVariableOp?
model/HidD1/MatMulMatMul model/flatten_9/Reshape:output:0)model/HidD1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidD1/MatMul?
"model/HidD1/BiasAdd/ReadVariableOpReadVariableOp+model_hidd1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"model/HidD1/BiasAdd/ReadVariableOp?
model/HidD1/BiasAddBiasAddmodel/HidD1/MatMul:product:0*model/HidD1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidD1/BiasAdd}
model/HidD1/ReluRelumodel/HidD1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidD1/Relu?
!model/HidE1/MatMul/ReadVariableOpReadVariableOp*model_hide1_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02#
!model/HidE1/MatMul/ReadVariableOp?
model/HidE1/MatMulMatMul!model/flatten_10/Reshape:output:0)model/HidE1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidE1/MatMul?
"model/HidE1/BiasAdd/ReadVariableOpReadVariableOp+model_hide1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"model/HidE1/BiasAdd/ReadVariableOp?
model/HidE1/BiasAddBiasAddmodel/HidE1/MatMul:product:0*model/HidE1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidE1/BiasAdd}
model/HidE1/ReluRelumodel/HidE1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidE1/Relu?
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axis?
model/concatenate/concatConcatV2model/HidA/Relu:activations:0model/HidA2/Relu:activations:0model/HidA3/Relu:activations:0model/HidA4/Relu:activations:0model/HidB/Relu:activations:0model/HidB2/Relu:activations:0model/HidB3/Relu:activations:0model/HidC/Relu:activations:0model/HidC2/Relu:activations:0model/HidD1/Relu:activations:0model/HidE1/Relu:activations:0&model/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
model/concatenate/concat?
 model/Hid1/MatMul/ReadVariableOpReadVariableOp)model_hid1_matmul_readvariableop_resource* 
_output_shapes
:
??*
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
_output_shapes
:
??*
dtype02$
"model/output/MatMul/ReadVariableOp?
model/output/MatMulMatMulmodel/Hid2/Tanh:y:0*model/output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/output/MatMul?
#model/output/BiasAdd/ReadVariableOpReadVariableOp,model_output_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#model/output/BiasAdd/ReadVariableOp?
model/output/BiasAddBiasAddmodel/output/MatMul:product:0+model/output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/output/BiasAdd?
model/output/SoftmaxSoftmaxmodel/output/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/output/Softmaxz
IdentityIdentitymodel/output/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp"^model/ConA/BiasAdd/ReadVariableOp!^model/ConA/Conv2D/ReadVariableOp#^model/ConA2/BiasAdd/ReadVariableOp"^model/ConA2/Conv2D/ReadVariableOp#^model/ConA3/BiasAdd/ReadVariableOp"^model/ConA3/Conv2D/ReadVariableOp#^model/ConA4/BiasAdd/ReadVariableOp"^model/ConA4/Conv2D/ReadVariableOp"^model/ConB/BiasAdd/ReadVariableOp!^model/ConB/Conv2D/ReadVariableOp#^model/ConB2/BiasAdd/ReadVariableOp"^model/ConB2/Conv2D/ReadVariableOp#^model/ConB3/BiasAdd/ReadVariableOp"^model/ConB3/Conv2D/ReadVariableOp"^model/ConC/BiasAdd/ReadVariableOp!^model/ConC/Conv2D/ReadVariableOp#^model/ConC2/BiasAdd/ReadVariableOp"^model/ConC2/Conv2D/ReadVariableOp"^model/ConD/BiasAdd/ReadVariableOp!^model/ConD/Conv2D/ReadVariableOp"^model/ConE/BiasAdd/ReadVariableOp!^model/ConE/Conv2D/ReadVariableOp"^model/Hid1/BiasAdd/ReadVariableOp!^model/Hid1/MatMul/ReadVariableOp"^model/Hid2/BiasAdd/ReadVariableOp!^model/Hid2/MatMul/ReadVariableOp"^model/HidA/BiasAdd/ReadVariableOp!^model/HidA/MatMul/ReadVariableOp#^model/HidA2/BiasAdd/ReadVariableOp"^model/HidA2/MatMul/ReadVariableOp#^model/HidA3/BiasAdd/ReadVariableOp"^model/HidA3/MatMul/ReadVariableOp#^model/HidA4/BiasAdd/ReadVariableOp"^model/HidA4/MatMul/ReadVariableOp"^model/HidB/BiasAdd/ReadVariableOp!^model/HidB/MatMul/ReadVariableOp#^model/HidB2/BiasAdd/ReadVariableOp"^model/HidB2/MatMul/ReadVariableOp#^model/HidB3/BiasAdd/ReadVariableOp"^model/HidB3/MatMul/ReadVariableOp"^model/HidC/BiasAdd/ReadVariableOp!^model/HidC/MatMul/ReadVariableOp#^model/HidC2/BiasAdd/ReadVariableOp"^model/HidC2/MatMul/ReadVariableOp#^model/HidD1/BiasAdd/ReadVariableOp"^model/HidD1/MatMul/ReadVariableOp#^model/HidE1/BiasAdd/ReadVariableOp"^model/HidE1/MatMul/ReadVariableOp$^model/output/BiasAdd/ReadVariableOp#^model/output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!model/ConA/BiasAdd/ReadVariableOp!model/ConA/BiasAdd/ReadVariableOp2D
 model/ConA/Conv2D/ReadVariableOp model/ConA/Conv2D/ReadVariableOp2H
"model/ConA2/BiasAdd/ReadVariableOp"model/ConA2/BiasAdd/ReadVariableOp2F
!model/ConA2/Conv2D/ReadVariableOp!model/ConA2/Conv2D/ReadVariableOp2H
"model/ConA3/BiasAdd/ReadVariableOp"model/ConA3/BiasAdd/ReadVariableOp2F
!model/ConA3/Conv2D/ReadVariableOp!model/ConA3/Conv2D/ReadVariableOp2H
"model/ConA4/BiasAdd/ReadVariableOp"model/ConA4/BiasAdd/ReadVariableOp2F
!model/ConA4/Conv2D/ReadVariableOp!model/ConA4/Conv2D/ReadVariableOp2F
!model/ConB/BiasAdd/ReadVariableOp!model/ConB/BiasAdd/ReadVariableOp2D
 model/ConB/Conv2D/ReadVariableOp model/ConB/Conv2D/ReadVariableOp2H
"model/ConB2/BiasAdd/ReadVariableOp"model/ConB2/BiasAdd/ReadVariableOp2F
!model/ConB2/Conv2D/ReadVariableOp!model/ConB2/Conv2D/ReadVariableOp2H
"model/ConB3/BiasAdd/ReadVariableOp"model/ConB3/BiasAdd/ReadVariableOp2F
!model/ConB3/Conv2D/ReadVariableOp!model/ConB3/Conv2D/ReadVariableOp2F
!model/ConC/BiasAdd/ReadVariableOp!model/ConC/BiasAdd/ReadVariableOp2D
 model/ConC/Conv2D/ReadVariableOp model/ConC/Conv2D/ReadVariableOp2H
"model/ConC2/BiasAdd/ReadVariableOp"model/ConC2/BiasAdd/ReadVariableOp2F
!model/ConC2/Conv2D/ReadVariableOp!model/ConC2/Conv2D/ReadVariableOp2F
!model/ConD/BiasAdd/ReadVariableOp!model/ConD/BiasAdd/ReadVariableOp2D
 model/ConD/Conv2D/ReadVariableOp model/ConD/Conv2D/ReadVariableOp2F
!model/ConE/BiasAdd/ReadVariableOp!model/ConE/BiasAdd/ReadVariableOp2D
 model/ConE/Conv2D/ReadVariableOp model/ConE/Conv2D/ReadVariableOp2F
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
!model/HidB/BiasAdd/ReadVariableOp!model/HidB/BiasAdd/ReadVariableOp2D
 model/HidB/MatMul/ReadVariableOp model/HidB/MatMul/ReadVariableOp2H
"model/HidB2/BiasAdd/ReadVariableOp"model/HidB2/BiasAdd/ReadVariableOp2F
!model/HidB2/MatMul/ReadVariableOp!model/HidB2/MatMul/ReadVariableOp2H
"model/HidB3/BiasAdd/ReadVariableOp"model/HidB3/BiasAdd/ReadVariableOp2F
!model/HidB3/MatMul/ReadVariableOp!model/HidB3/MatMul/ReadVariableOp2F
!model/HidC/BiasAdd/ReadVariableOp!model/HidC/BiasAdd/ReadVariableOp2D
 model/HidC/MatMul/ReadVariableOp model/HidC/MatMul/ReadVariableOp2H
"model/HidC2/BiasAdd/ReadVariableOp"model/HidC2/BiasAdd/ReadVariableOp2F
!model/HidC2/MatMul/ReadVariableOp!model/HidC2/MatMul/ReadVariableOp2H
"model/HidD1/BiasAdd/ReadVariableOp"model/HidD1/BiasAdd/ReadVariableOp2F
!model/HidD1/MatMul/ReadVariableOp!model/HidD1/MatMul/ReadVariableOp2H
"model/HidE1/BiasAdd/ReadVariableOp"model/HidE1/BiasAdd/ReadVariableOp2F
!model/HidE1/MatMul/ReadVariableOp!model/HidE1/MatMul/ReadVariableOp2J
#model/output/BiasAdd/ReadVariableOp#model/output/BiasAdd/ReadVariableOp2H
"model/output/MatMul/ReadVariableOp"model/output/MatMul/ReadVariableOp:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
]
A__inference_PoolE_layer_call_and_return_conditional_losses_191285

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????D; *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????D; 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
&__inference_HidB3_layer_call_fn_195107

inputs
unknown:
?F?
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
A__inference_HidB3_layer_call_and_return_conditional_losses_1915682
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
:??????????F: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????F
 
_user_specified_nameinputs
?
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_191316

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????D; 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????D; 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????D; :W S
/
_output_shapes
:?????????D; 
 
_user_specified_nameinputs
?
?
A__inference_HidA2_layer_call_and_return_conditional_losses_191483

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
E__inference_dropout_6_layer_call_and_return_conditional_losses_194662

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????(2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????(2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
%__inference_HidB_layer_call_fn_195067

inputs
unknown:ȓ?
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
@__inference_HidB_layer_call_and_return_conditional_losses_1915342
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
:?????????ȓ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:?????????ȓ
 
_user_specified_nameinputs
?
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_191365

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
?
?
,__inference_concatenate_layer_call_fn_195218
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
	inputs_10
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_1916582
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:R N
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
_user_specified_name	inputs/10
?
?
@__inference_HidA_layer_call_and_return_conditional_losses_191466

inputs3
matmul_readvariableop_resource:???.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
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
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_192195

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
:?????????G=2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????G=*
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
:?????????G=2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????G=2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????G=2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????G=2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????G=:W S
/
_output_shapes
:?????????G=
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_194109

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:(

unknown_10:($

unknown_11: 

unknown_12: $

unknown_13:	

unknown_14:$

unknown_15:

unknown_16:$

unknown_17:(

unknown_18:($

unknown_19:((

unknown_20:(

unknown_21:???

unknown_22:	?

unknown_23:???

unknown_24:	?

unknown_25:
?F?

unknown_26:	?

unknown_27:
??

unknown_28:	?

unknown_29:ȓ?

unknown_30:	?

unknown_31:???

unknown_32:	?

unknown_33:
?F?

unknown_34:	?

unknown_35:???

unknown_36:	?

unknown_37:???

unknown_38:	?

unknown_39:???

unknown_40:	?

unknown_41:???

unknown_42:	?

unknown_43:
??

unknown_44:	?

unknown_45:
??

unknown_46:	?

unknown_47:
??

unknown_48:	?
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
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1927252
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_191815	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:(

unknown_10:($

unknown_11: 

unknown_12: $

unknown_13:	

unknown_14:$

unknown_15:

unknown_16:$

unknown_17:(

unknown_18:($

unknown_19:((

unknown_20:(

unknown_21:???

unknown_22:	?

unknown_23:???

unknown_24:	?

unknown_25:
?F?

unknown_26:	?

unknown_27:
??

unknown_28:	?

unknown_29:ȓ?

unknown_30:	?

unknown_31:???

unknown_32:	?

unknown_33:
?F?

unknown_34:	?

unknown_35:???

unknown_36:	?

unknown_37:???

unknown_38:	?

unknown_39:???

unknown_40:	?

unknown_41:???

unknown_42:	?

unknown_43:
??

unknown_44:	?

unknown_45:
??

unknown_46:	?

unknown_47:
??

unknown_48:	?
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
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1917122
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
^
B__inference_PoolB3_layer_call_and_return_conditional_losses_194555

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
E__inference_flatten_5_layer_call_and_return_conditional_losses_191413

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
?
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_192379

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
:?????????H?2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????H?*
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
:?????????H?2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????H?2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????H?2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????H?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????H?:W S
/
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?
?
@__inference_HidB_layer_call_and_return_conditional_losses_191534

inputs3
matmul_readvariableop_resource:ȓ?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:ȓ?*
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
:?????????ȓ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:?????????ȓ
 
_user_specified_nameinputs
?
?
&__inference_HidA4_layer_call_fn_195047

inputs
unknown:
??
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
A__inference_HidA4_layer_call_and_return_conditional_losses_1915172
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
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
^
B__inference_PoolA3_layer_call_and_return_conditional_losses_194373

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????(*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????!(:W S
/
_output_shapes
:?????????!(
 
_user_specified_nameinputs
?
C
'__inference_PoolB3_layer_call_fn_194565

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
B__inference_PoolB3_layer_call_and_return_conditional_losses_1909512
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
E__inference_dropout_8_layer_call_and_return_conditional_losses_191344

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
?
d
+__inference_dropout_10_layer_call_fn_194792

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
:?????????"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_1921722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????"2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????"22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????"
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_194004

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:(

unknown_10:($

unknown_11: 

unknown_12: $

unknown_13:	

unknown_14:$

unknown_15:

unknown_16:$

unknown_17:(

unknown_18:($

unknown_19:((

unknown_20:(

unknown_21:???

unknown_22:	?

unknown_23:???

unknown_24:	?

unknown_25:
?F?

unknown_26:	?

unknown_27:
??

unknown_28:	?

unknown_29:ȓ?

unknown_30:	?

unknown_31:???

unknown_32:	?

unknown_33:
?F?

unknown_34:	?

unknown_35:???

unknown_36:	?

unknown_37:???

unknown_38:	?

unknown_39:???

unknown_40:	?

unknown_41:???

unknown_42:	?

unknown_43:
??

unknown_44:	?

unknown_45:
??

unknown_46:	?

unknown_47:
??

unknown_48:	?
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
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1917122
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
C
'__inference_PoolC2_layer_call_fn_194590

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
:?????????"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolC2_layer_call_and_return_conditional_losses_1912972
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????E<:W S
/
_output_shapes
:?????????E<
 
_user_specified_nameinputs
?
F
*__inference_flatten_1_layer_call_fn_194868

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
E__inference_flatten_1_layer_call_and_return_conditional_losses_1914452
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
?
]
A__inference_PoolA_layer_call_and_return_conditional_losses_191058

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????I?*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????I?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
^
B__inference_PoolA4_layer_call_and_return_conditional_losses_191309

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????(*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
%__inference_HidC_layer_call_fn_195127

inputs
unknown:???
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
@__inference_HidC_layer_call_and_return_conditional_losses_1915852
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
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_4_layer_call_and_return_conditional_losses_191421

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????	 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:?????????ȓ2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:?????????ȓ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????H?:W S
/
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?
^
B__inference_PoolA2_layer_call_and_return_conditional_losses_191135

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
A__inference_HidB3_layer_call_and_return_conditional_losses_191568

inputs2
matmul_readvariableop_resource:
?F?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?F?*
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
:??????????F: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????F
 
_user_specified_nameinputs
?
?
&__inference_ConB2_layer_call_fn_194343

inputs!
unknown:
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
:?????????F=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConB2_layer_call_and_return_conditional_losses_1911542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????F=2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????H?: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?
?
@__inference_ConC_layer_call_and_return_conditional_losses_194294

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
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
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????2

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
^
B__inference_PoolA4_layer_call_and_return_conditional_losses_194535

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
]
A__inference_PoolA_layer_call_and_return_conditional_losses_194134

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
?
&__inference_HidA3_layer_call_fn_195027

inputs
unknown:
?F?
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
A__inference_HidA3_layer_call_and_return_conditional_losses_1915002
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
:??????????F: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????F
 
_user_specified_nameinputs
?
F
*__inference_dropout_9_layer_call_fn_194760

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
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_1913372
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
%__inference_ConC_layer_call_fn_194303

inputs!
unknown:	
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConC_layer_call_and_return_conditional_losses_1911182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

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
?
B
&__inference_PoolD_layer_call_fn_194605

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
A__inference_PoolD_layer_call_and_return_conditional_losses_1909952
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
E__inference_flatten_1_layer_call_and_return_conditional_losses_191445

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
?
^
B__inference_PoolB2_layer_call_and_return_conditional_losses_191188

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
:?????????F=:W S
/
_output_shapes
:?????????F=
 
_user_specified_nameinputs
?
C
'__inference_PoolC2_layer_call_fn_194585

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
B__inference_PoolC2_layer_call_and_return_conditional_losses_1909732
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
@__inference_HidC_layer_call_and_return_conditional_losses_195118

inputs3
matmul_readvariableop_resource:???.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
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
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
]
A__inference_PoolC_layer_call_and_return_conditional_losses_191141

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????G>*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????G>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
A__inference_model_layer_call_and_return_conditional_losses_191712

inputs%
cona_191049:
cona_191051:%
conb_191072:
conb_191074:&
cona2_191102:
cona2_191104:%
conc_191119:	
conc_191121:&
conb2_191155:
conb2_191157:&
cona3_191172:(
cona3_191174:(%
cone_191208: 
cone_191210: %
cond_191225:	
cond_191227:&
conc2_191242:
conc2_191244:&
conb3_191259:(
conb3_191261:(&
cona4_191276:((
cona4_191278:( 
hida_191467:???
hida_191469:	?!
hida2_191484:???
hida2_191486:	? 
hida3_191501:
?F?
hida3_191503:	? 
hida4_191518:
??
hida4_191520:	? 
hidb_191535:ȓ?
hidb_191537:	?!
hidb2_191552:???
hidb2_191554:	? 
hidb3_191569:
?F?
hidb3_191571:	? 
hidc_191586:???
hidc_191588:	?!
hidc2_191603:???
hidc2_191605:	?!
hidd1_191620:???
hidd1_191622:	?!
hide1_191637:???
hide1_191639:	?
hid1_191672:
??
hid1_191674:	?
hid2_191689:
??
hid2_191691:	?!
output_191706:
??
output_191708:	?
identity??ConA/StatefulPartitionedCall?ConA2/StatefulPartitionedCall?ConA3/StatefulPartitionedCall?ConA4/StatefulPartitionedCall?ConB/StatefulPartitionedCall?ConB2/StatefulPartitionedCall?ConB3/StatefulPartitionedCall?ConC/StatefulPartitionedCall?ConC2/StatefulPartitionedCall?ConD/StatefulPartitionedCall?ConE/StatefulPartitionedCall?Hid1/StatefulPartitionedCall?Hid2/StatefulPartitionedCall?HidA/StatefulPartitionedCall?HidA2/StatefulPartitionedCall?HidA3/StatefulPartitionedCall?HidA4/StatefulPartitionedCall?HidB/StatefulPartitionedCall?HidB2/StatefulPartitionedCall?HidB3/StatefulPartitionedCall?HidC/StatefulPartitionedCall?HidC2/StatefulPartitionedCall?HidD1/StatefulPartitionedCall?HidE1/StatefulPartitionedCall?output/StatefulPartitionedCall?
ConA/StatefulPartitionedCallStatefulPartitionedCallinputscona_191049cona_191051*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConA_layer_call_and_return_conditional_losses_1910482
ConA/StatefulPartitionedCall?
PoolA/PartitionedCallPartitionedCall%ConA/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolA_layer_call_and_return_conditional_losses_1910582
PoolA/PartitionedCall?
ConB/StatefulPartitionedCallStatefulPartitionedCallinputsconb_191072conb_191074*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConB_layer_call_and_return_conditional_losses_1910712
ConB/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCallPoolA/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1910822
dropout/PartitionedCall?
PoolB/PartitionedCallPartitionedCall%ConB/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????H?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolB_layer_call_and_return_conditional_losses_1910882
PoolB/PartitionedCall?
ConA2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0cona2_191102cona2_191104*
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
A__inference_ConA2_layer_call_and_return_conditional_losses_1911012
ConA2/StatefulPartitionedCall?
ConC/StatefulPartitionedCallStatefulPartitionedCallinputsconc_191119conc_191121*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConC_layer_call_and_return_conditional_losses_1911182
ConC/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCallPoolB/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????H?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1911292
dropout_1/PartitionedCall?
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
B__inference_PoolA2_layer_call_and_return_conditional_losses_1911352
PoolA2/PartitionedCall?
PoolC/PartitionedCallPartitionedCall%ConC/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????G>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolC_layer_call_and_return_conditional_losses_1911412
PoolC/PartitionedCall?
ConB2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conb2_191155conb2_191157*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConB2_layer_call_and_return_conditional_losses_1911542
ConB2/StatefulPartitionedCall?
ConA3/StatefulPartitionedCallStatefulPartitionedCallPoolA2/PartitionedCall:output:0cona3_191172cona3_191174*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConA3_layer_call_and_return_conditional_losses_1911712
ConA3/StatefulPartitionedCall?
dropout_2/PartitionedCallPartitionedCallPoolC/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????G>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_1911822
dropout_2/PartitionedCall?
PoolB2/PartitionedCallPartitionedCall&ConB2/StatefulPartitionedCall:output:0*
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
B__inference_PoolB2_layer_call_and_return_conditional_losses_1911882
PoolB2/PartitionedCall?
PoolA3/PartitionedCallPartitionedCall&ConA3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA3_layer_call_and_return_conditional_losses_1911942
PoolA3/PartitionedCall?
ConE/StatefulPartitionedCallStatefulPartitionedCallinputscone_191208cone_191210*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConE_layer_call_and_return_conditional_losses_1912072
ConE/StatefulPartitionedCall?
ConD/StatefulPartitionedCallStatefulPartitionedCallinputscond_191225cond_191227*
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
@__inference_ConD_layer_call_and_return_conditional_losses_1912242
ConD/StatefulPartitionedCall?
ConC2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0conc2_191242conc2_191244*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????E<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConC2_layer_call_and_return_conditional_losses_1912412
ConC2/StatefulPartitionedCall?
ConB3/StatefulPartitionedCallStatefulPartitionedCallPoolB2/PartitionedCall:output:0conb3_191259conb3_191261*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConB3_layer_call_and_return_conditional_losses_1912582
ConB3/StatefulPartitionedCall?
ConA4/StatefulPartitionedCallStatefulPartitionedCallPoolA3/PartitionedCall:output:0cona4_191276cona4_191278*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConA4_layer_call_and_return_conditional_losses_1912752
ConA4/StatefulPartitionedCall?
PoolE/PartitionedCallPartitionedCall%ConE/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????D; * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolE_layer_call_and_return_conditional_losses_1912852
PoolE/PartitionedCall?
PoolD/PartitionedCallPartitionedCall%ConD/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????G=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolD_layer_call_and_return_conditional_losses_1912912
PoolD/PartitionedCall?
PoolC2/PartitionedCallPartitionedCall&ConC2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolC2_layer_call_and_return_conditional_losses_1912972
PoolC2/PartitionedCall?
PoolB3/PartitionedCallPartitionedCall&ConB3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolB3_layer_call_and_return_conditional_losses_1913032
PoolB3/PartitionedCall?
PoolA4/PartitionedCallPartitionedCall&ConA4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA4_layer_call_and_return_conditional_losses_1913092
PoolA4/PartitionedCall?
dropout_4/PartitionedCallPartitionedCallPoolE/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????D; * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1913162
dropout_4/PartitionedCall?
dropout_3/PartitionedCallPartitionedCallPoolD/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????G=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_1913232
dropout_3/PartitionedCall?
dropout_10/PartitionedCallPartitionedCallPoolC2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_1913302
dropout_10/PartitionedCall?
dropout_9/PartitionedCallPartitionedCallPoolB3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_1913372
dropout_9/PartitionedCall?
dropout_8/PartitionedCallPartitionedCallPoolB2/PartitionedCall:output:0*
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
E__inference_dropout_8_layer_call_and_return_conditional_losses_1913442
dropout_8/PartitionedCall?
dropout_7/PartitionedCallPartitionedCallPoolA4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_1913512
dropout_7/PartitionedCall?
dropout_6/PartitionedCallPartitionedCallPoolA3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_1913582
dropout_6/PartitionedCall?
dropout_5/PartitionedCallPartitionedCallPoolA2/PartitionedCall:output:0*
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
E__inference_dropout_5_layer_call_and_return_conditional_losses_1913652
dropout_5/PartitionedCall?
flatten_10/PartitionedCallPartitionedCall"dropout_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_10_layer_call_and_return_conditional_losses_1913732
flatten_10/PartitionedCall?
flatten_9/PartitionedCallPartitionedCall"dropout_3/PartitionedCall:output:0*
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
GPU2*0J 8? *N
fIRG
E__inference_flatten_9_layer_call_and_return_conditional_losses_1913812
flatten_9/PartitionedCall?
flatten_8/PartitionedCallPartitionedCall#dropout_10/PartitionedCall:output:0*
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
E__inference_flatten_8_layer_call_and_return_conditional_losses_1913892
flatten_8/PartitionedCall?
flatten_7/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_1913972
flatten_7/PartitionedCall?
flatten_6/PartitionedCallPartitionedCall"dropout_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????F* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_1914052
flatten_6/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall"dropout_8/PartitionedCall:output:0*
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
E__inference_flatten_5_layer_call_and_return_conditional_losses_1914132
flatten_5/PartitionedCall?
flatten_4/PartitionedCallPartitionedCallPoolB/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????ȓ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_1914212
flatten_4/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall"dropout_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1914292
flatten_3/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall"dropout_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????F* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_1914372
flatten_2/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall"dropout_5/PartitionedCall:output:0*
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
E__inference_flatten_1_layer_call_and_return_conditional_losses_1914452
flatten_1/PartitionedCall?
flatten/PartitionedCallPartitionedCallPoolA/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1914532
flatten/PartitionedCall?
HidA/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0hida_191467hida_191469*
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
@__inference_HidA_layer_call_and_return_conditional_losses_1914662
HidA/StatefulPartitionedCall?
HidA2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0hida2_191484hida2_191486*
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
A__inference_HidA2_layer_call_and_return_conditional_losses_1914832
HidA2/StatefulPartitionedCall?
HidA3/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0hida3_191501hida3_191503*
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
A__inference_HidA3_layer_call_and_return_conditional_losses_1915002
HidA3/StatefulPartitionedCall?
HidA4/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0hida4_191518hida4_191520*
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
A__inference_HidA4_layer_call_and_return_conditional_losses_1915172
HidA4/StatefulPartitionedCall?
HidB/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0hidb_191535hidb_191537*
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
@__inference_HidB_layer_call_and_return_conditional_losses_1915342
HidB/StatefulPartitionedCall?
HidB2/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0hidb2_191552hidb2_191554*
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
A__inference_HidB2_layer_call_and_return_conditional_losses_1915512
HidB2/StatefulPartitionedCall?
HidB3/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0hidb3_191569hidb3_191571*
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
A__inference_HidB3_layer_call_and_return_conditional_losses_1915682
HidB3/StatefulPartitionedCall?
HidC/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0hidc_191586hidc_191588*
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
@__inference_HidC_layer_call_and_return_conditional_losses_1915852
HidC/StatefulPartitionedCall?
HidC2/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0hidc2_191603hidc2_191605*
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
A__inference_HidC2_layer_call_and_return_conditional_losses_1916022
HidC2/StatefulPartitionedCall?
HidD1/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0hidd1_191620hidd1_191622*
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
A__inference_HidD1_layer_call_and_return_conditional_losses_1916192
HidD1/StatefulPartitionedCall?
HidE1/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0hide1_191637hide1_191639*
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
A__inference_HidE1_layer_call_and_return_conditional_losses_1916362
HidE1/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall%HidA/StatefulPartitionedCall:output:0&HidA2/StatefulPartitionedCall:output:0&HidA3/StatefulPartitionedCall:output:0&HidA4/StatefulPartitionedCall:output:0%HidB/StatefulPartitionedCall:output:0&HidB2/StatefulPartitionedCall:output:0&HidB3/StatefulPartitionedCall:output:0%HidC/StatefulPartitionedCall:output:0&HidC2/StatefulPartitionedCall:output:0&HidD1/StatefulPartitionedCall:output:0&HidE1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_1916582
concatenate/PartitionedCall?
Hid1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0hid1_191672hid1_191674*
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
@__inference_Hid1_layer_call_and_return_conditional_losses_1916712
Hid1/StatefulPartitionedCall?
Hid2/StatefulPartitionedCallStatefulPartitionedCall%Hid1/StatefulPartitionedCall:output:0hid2_191689hid2_191691*
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
@__inference_Hid2_layer_call_and_return_conditional_losses_1916882
Hid2/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall%Hid2/StatefulPartitionedCall:output:0output_191706output_191708*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1917052 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^ConA/StatefulPartitionedCall^ConA2/StatefulPartitionedCall^ConA3/StatefulPartitionedCall^ConA4/StatefulPartitionedCall^ConB/StatefulPartitionedCall^ConB2/StatefulPartitionedCall^ConB3/StatefulPartitionedCall^ConC/StatefulPartitionedCall^ConC2/StatefulPartitionedCall^ConD/StatefulPartitionedCall^ConE/StatefulPartitionedCall^Hid1/StatefulPartitionedCall^Hid2/StatefulPartitionedCall^HidA/StatefulPartitionedCall^HidA2/StatefulPartitionedCall^HidA3/StatefulPartitionedCall^HidA4/StatefulPartitionedCall^HidB/StatefulPartitionedCall^HidB2/StatefulPartitionedCall^HidB3/StatefulPartitionedCall^HidC/StatefulPartitionedCall^HidC2/StatefulPartitionedCall^HidD1/StatefulPartitionedCall^HidE1/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
ConA/StatefulPartitionedCallConA/StatefulPartitionedCall2>
ConA2/StatefulPartitionedCallConA2/StatefulPartitionedCall2>
ConA3/StatefulPartitionedCallConA3/StatefulPartitionedCall2>
ConA4/StatefulPartitionedCallConA4/StatefulPartitionedCall2<
ConB/StatefulPartitionedCallConB/StatefulPartitionedCall2>
ConB2/StatefulPartitionedCallConB2/StatefulPartitionedCall2>
ConB3/StatefulPartitionedCallConB3/StatefulPartitionedCall2<
ConC/StatefulPartitionedCallConC/StatefulPartitionedCall2>
ConC2/StatefulPartitionedCallConC2/StatefulPartitionedCall2<
ConD/StatefulPartitionedCallConD/StatefulPartitionedCall2<
ConE/StatefulPartitionedCallConE/StatefulPartitionedCall2<
Hid1/StatefulPartitionedCallHid1/StatefulPartitionedCall2<
Hid2/StatefulPartitionedCallHid2/StatefulPartitionedCall2<
HidA/StatefulPartitionedCallHidA/StatefulPartitionedCall2>
HidA2/StatefulPartitionedCallHidA2/StatefulPartitionedCall2>
HidA3/StatefulPartitionedCallHidA3/StatefulPartitionedCall2>
HidA4/StatefulPartitionedCallHidA4/StatefulPartitionedCall2<
HidB/StatefulPartitionedCallHidB/StatefulPartitionedCall2>
HidB2/StatefulPartitionedCallHidB2/StatefulPartitionedCall2>
HidB3/StatefulPartitionedCallHidB3/StatefulPartitionedCall2<
HidC/StatefulPartitionedCallHidC/StatefulPartitionedCall2>
HidC2/StatefulPartitionedCallHidC2/StatefulPartitionedCall2>
HidD1/StatefulPartitionedCallHidD1/StatefulPartitionedCall2>
HidE1/StatefulPartitionedCallHidE1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
]
A__inference_PoolD_layer_call_and_return_conditional_losses_194600

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????G=*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????G=2

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
d
E__inference_dropout_6_layer_call_and_return_conditional_losses_192080

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
:?????????(2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????(2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????(2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
B
&__inference_PoolB_layer_call_fn_194231

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
A__inference_PoolB_layer_call_and_return_conditional_losses_1908192
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
A__inference_ConC2_layer_call_and_return_conditional_losses_194481

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
:?????????E<*
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
:?????????E<2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????E<2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????E<2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????G>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????G>
 
_user_specified_nameinputs
?
b
F__inference_flatten_10_layer_call_and_return_conditional_losses_191373

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????? 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????D; :W S
/
_output_shapes
:?????????D; 
 
_user_specified_nameinputs
?
d
F__inference_dropout_10_layer_call_and_return_conditional_losses_191330

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????"2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????"2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????":W S
/
_output_shapes
:?????????"
 
_user_specified_nameinputs
??
?
A__inference_model_layer_call_and_return_conditional_losses_193259	
input%
cona_193099:
cona_193101:%
conb_193105:
conb_193107:&
cona2_193112:
cona2_193114:%
conc_193117:	
conc_193119:&
conb2_193125:
conb2_193127:&
cona3_193130:(
cona3_193132:(%
cone_193138: 
cone_193140: %
cond_193143:	
cond_193145:&
conc2_193148:
conc2_193150:&
conb3_193153:(
conb3_193155:(&
cona4_193158:((
cona4_193160:( 
hida_193187:???
hida_193189:	?!
hida2_193192:???
hida2_193194:	? 
hida3_193197:
?F?
hida3_193199:	? 
hida4_193202:
??
hida4_193204:	? 
hidb_193207:ȓ?
hidb_193209:	?!
hidb2_193212:???
hidb2_193214:	? 
hidb3_193217:
?F?
hidb3_193219:	? 
hidc_193222:???
hidc_193224:	?!
hidc2_193227:???
hidc2_193229:	?!
hidd1_193232:???
hidd1_193234:	?!
hide1_193237:???
hide1_193239:	?
hid1_193243:
??
hid1_193245:	?
hid2_193248:
??
hid2_193250:	?!
output_193253:
??
output_193255:	?
identity??ConA/StatefulPartitionedCall?ConA2/StatefulPartitionedCall?ConA3/StatefulPartitionedCall?ConA4/StatefulPartitionedCall?ConB/StatefulPartitionedCall?ConB2/StatefulPartitionedCall?ConB3/StatefulPartitionedCall?ConC/StatefulPartitionedCall?ConC2/StatefulPartitionedCall?ConD/StatefulPartitionedCall?ConE/StatefulPartitionedCall?Hid1/StatefulPartitionedCall?Hid2/StatefulPartitionedCall?HidA/StatefulPartitionedCall?HidA2/StatefulPartitionedCall?HidA3/StatefulPartitionedCall?HidA4/StatefulPartitionedCall?HidB/StatefulPartitionedCall?HidB2/StatefulPartitionedCall?HidB3/StatefulPartitionedCall?HidC/StatefulPartitionedCall?HidC2/StatefulPartitionedCall?HidD1/StatefulPartitionedCall?HidE1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?"dropout_10/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?!dropout_5/StatefulPartitionedCall?!dropout_6/StatefulPartitionedCall?!dropout_7/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?!dropout_9/StatefulPartitionedCall?output/StatefulPartitionedCall?
ConA/StatefulPartitionedCallStatefulPartitionedCallinputcona_193099cona_193101*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConA_layer_call_and_return_conditional_losses_1910482
ConA/StatefulPartitionedCall?
PoolA/PartitionedCallPartitionedCall%ConA/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolA_layer_call_and_return_conditional_losses_1910582
PoolA/PartitionedCall?
ConB/StatefulPartitionedCallStatefulPartitionedCallinputconb_193105conb_193107*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConB_layer_call_and_return_conditional_losses_1910712
ConB/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCallPoolA/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1924272!
dropout/StatefulPartitionedCall?
PoolB/PartitionedCallPartitionedCall%ConB/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????H?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolB_layer_call_and_return_conditional_losses_1910882
PoolB/PartitionedCall?
ConA2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0cona2_193112cona2_193114*
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
A__inference_ConA2_layer_call_and_return_conditional_losses_1911012
ConA2/StatefulPartitionedCall?
ConC/StatefulPartitionedCallStatefulPartitionedCallinputconc_193117conc_193119*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConC_layer_call_and_return_conditional_losses_1911182
ConC/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCallPoolB/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????H?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1923792#
!dropout_1/StatefulPartitionedCall?
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
B__inference_PoolA2_layer_call_and_return_conditional_losses_1911352
PoolA2/PartitionedCall?
PoolC/PartitionedCallPartitionedCall%ConC/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????G>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolC_layer_call_and_return_conditional_losses_1911412
PoolC/PartitionedCall?
ConB2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conb2_193125conb2_193127*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConB2_layer_call_and_return_conditional_losses_1911542
ConB2/StatefulPartitionedCall?
ConA3/StatefulPartitionedCallStatefulPartitionedCallPoolA2/PartitionedCall:output:0cona3_193130cona3_193132*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConA3_layer_call_and_return_conditional_losses_1911712
ConA3/StatefulPartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCallPoolC/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????G>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_1923262#
!dropout_2/StatefulPartitionedCall?
PoolB2/PartitionedCallPartitionedCall&ConB2/StatefulPartitionedCall:output:0*
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
B__inference_PoolB2_layer_call_and_return_conditional_losses_1911882
PoolB2/PartitionedCall?
PoolA3/PartitionedCallPartitionedCall&ConA3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA3_layer_call_and_return_conditional_losses_1911942
PoolA3/PartitionedCall?
ConE/StatefulPartitionedCallStatefulPartitionedCallinputcone_193138cone_193140*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConE_layer_call_and_return_conditional_losses_1912072
ConE/StatefulPartitionedCall?
ConD/StatefulPartitionedCallStatefulPartitionedCallinputcond_193143cond_193145*
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
@__inference_ConD_layer_call_and_return_conditional_losses_1912242
ConD/StatefulPartitionedCall?
ConC2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0conc2_193148conc2_193150*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????E<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConC2_layer_call_and_return_conditional_losses_1912412
ConC2/StatefulPartitionedCall?
ConB3/StatefulPartitionedCallStatefulPartitionedCallPoolB2/PartitionedCall:output:0conb3_193153conb3_193155*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConB3_layer_call_and_return_conditional_losses_1912582
ConB3/StatefulPartitionedCall?
ConA4/StatefulPartitionedCallStatefulPartitionedCallPoolA3/PartitionedCall:output:0cona4_193158cona4_193160*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConA4_layer_call_and_return_conditional_losses_1912752
ConA4/StatefulPartitionedCall?
PoolE/PartitionedCallPartitionedCall%ConE/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????D; * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolE_layer_call_and_return_conditional_losses_1912852
PoolE/PartitionedCall?
PoolD/PartitionedCallPartitionedCall%ConD/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????G=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolD_layer_call_and_return_conditional_losses_1912912
PoolD/PartitionedCall?
PoolC2/PartitionedCallPartitionedCall&ConC2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolC2_layer_call_and_return_conditional_losses_1912972
PoolC2/PartitionedCall?
PoolB3/PartitionedCallPartitionedCall&ConB3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolB3_layer_call_and_return_conditional_losses_1913032
PoolB3/PartitionedCall?
PoolA4/PartitionedCallPartitionedCall&ConA4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA4_layer_call_and_return_conditional_losses_1913092
PoolA4/PartitionedCall?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCallPoolE/PartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????D; * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1922182#
!dropout_4/StatefulPartitionedCall?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCallPoolD/PartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????G=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_1921952#
!dropout_3/StatefulPartitionedCall?
"dropout_10/StatefulPartitionedCallStatefulPartitionedCallPoolC2/PartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_1921722$
"dropout_10/StatefulPartitionedCall?
!dropout_9/StatefulPartitionedCallStatefulPartitionedCallPoolB3/PartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_1921492#
!dropout_9/StatefulPartitionedCall?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCallPoolB2/PartitionedCall:output:0"^dropout_9/StatefulPartitionedCall*
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
E__inference_dropout_8_layer_call_and_return_conditional_losses_1921262#
!dropout_8/StatefulPartitionedCall?
!dropout_7/StatefulPartitionedCallStatefulPartitionedCallPoolA4/PartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_1921032#
!dropout_7/StatefulPartitionedCall?
!dropout_6/StatefulPartitionedCallStatefulPartitionedCallPoolA3/PartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_1920802#
!dropout_6/StatefulPartitionedCall?
!dropout_5/StatefulPartitionedCallStatefulPartitionedCallPoolA2/PartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
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
E__inference_dropout_5_layer_call_and_return_conditional_losses_1920572#
!dropout_5/StatefulPartitionedCall?
flatten_10/PartitionedCallPartitionedCall*dropout_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_10_layer_call_and_return_conditional_losses_1913732
flatten_10/PartitionedCall?
flatten_9/PartitionedCallPartitionedCall*dropout_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *N
fIRG
E__inference_flatten_9_layer_call_and_return_conditional_losses_1913812
flatten_9/PartitionedCall?
flatten_8/PartitionedCallPartitionedCall+dropout_10/StatefulPartitionedCall:output:0*
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
E__inference_flatten_8_layer_call_and_return_conditional_losses_1913892
flatten_8/PartitionedCall?
flatten_7/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_1913972
flatten_7/PartitionedCall?
flatten_6/PartitionedCallPartitionedCall*dropout_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????F* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_1914052
flatten_6/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall*dropout_8/StatefulPartitionedCall:output:0*
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
E__inference_flatten_5_layer_call_and_return_conditional_losses_1914132
flatten_5/PartitionedCall?
flatten_4/PartitionedCallPartitionedCallPoolB/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????ȓ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_1914212
flatten_4/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall*dropout_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1914292
flatten_3/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall*dropout_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????F* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_1914372
flatten_2/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall*dropout_5/StatefulPartitionedCall:output:0*
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
E__inference_flatten_1_layer_call_and_return_conditional_losses_1914452
flatten_1/PartitionedCall?
flatten/PartitionedCallPartitionedCallPoolA/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1914532
flatten/PartitionedCall?
HidA/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0hida_193187hida_193189*
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
@__inference_HidA_layer_call_and_return_conditional_losses_1914662
HidA/StatefulPartitionedCall?
HidA2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0hida2_193192hida2_193194*
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
A__inference_HidA2_layer_call_and_return_conditional_losses_1914832
HidA2/StatefulPartitionedCall?
HidA3/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0hida3_193197hida3_193199*
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
A__inference_HidA3_layer_call_and_return_conditional_losses_1915002
HidA3/StatefulPartitionedCall?
HidA4/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0hida4_193202hida4_193204*
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
A__inference_HidA4_layer_call_and_return_conditional_losses_1915172
HidA4/StatefulPartitionedCall?
HidB/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0hidb_193207hidb_193209*
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
@__inference_HidB_layer_call_and_return_conditional_losses_1915342
HidB/StatefulPartitionedCall?
HidB2/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0hidb2_193212hidb2_193214*
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
A__inference_HidB2_layer_call_and_return_conditional_losses_1915512
HidB2/StatefulPartitionedCall?
HidB3/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0hidb3_193217hidb3_193219*
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
A__inference_HidB3_layer_call_and_return_conditional_losses_1915682
HidB3/StatefulPartitionedCall?
HidC/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0hidc_193222hidc_193224*
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
@__inference_HidC_layer_call_and_return_conditional_losses_1915852
HidC/StatefulPartitionedCall?
HidC2/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0hidc2_193227hidc2_193229*
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
A__inference_HidC2_layer_call_and_return_conditional_losses_1916022
HidC2/StatefulPartitionedCall?
HidD1/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0hidd1_193232hidd1_193234*
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
A__inference_HidD1_layer_call_and_return_conditional_losses_1916192
HidD1/StatefulPartitionedCall?
HidE1/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0hide1_193237hide1_193239*
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
A__inference_HidE1_layer_call_and_return_conditional_losses_1916362
HidE1/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall%HidA/StatefulPartitionedCall:output:0&HidA2/StatefulPartitionedCall:output:0&HidA3/StatefulPartitionedCall:output:0&HidA4/StatefulPartitionedCall:output:0%HidB/StatefulPartitionedCall:output:0&HidB2/StatefulPartitionedCall:output:0&HidB3/StatefulPartitionedCall:output:0%HidC/StatefulPartitionedCall:output:0&HidC2/StatefulPartitionedCall:output:0&HidD1/StatefulPartitionedCall:output:0&HidE1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_1916582
concatenate/PartitionedCall?
Hid1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0hid1_193243hid1_193245*
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
@__inference_Hid1_layer_call_and_return_conditional_losses_1916712
Hid1/StatefulPartitionedCall?
Hid2/StatefulPartitionedCallStatefulPartitionedCall%Hid1/StatefulPartitionedCall:output:0hid2_193248hid2_193250*
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
@__inference_Hid2_layer_call_and_return_conditional_losses_1916882
Hid2/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall%Hid2/StatefulPartitionedCall:output:0output_193253output_193255*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1917052 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?	
NoOpNoOp^ConA/StatefulPartitionedCall^ConA2/StatefulPartitionedCall^ConA3/StatefulPartitionedCall^ConA4/StatefulPartitionedCall^ConB/StatefulPartitionedCall^ConB2/StatefulPartitionedCall^ConB3/StatefulPartitionedCall^ConC/StatefulPartitionedCall^ConC2/StatefulPartitionedCall^ConD/StatefulPartitionedCall^ConE/StatefulPartitionedCall^Hid1/StatefulPartitionedCall^Hid2/StatefulPartitionedCall^HidA/StatefulPartitionedCall^HidA2/StatefulPartitionedCall^HidA3/StatefulPartitionedCall^HidA4/StatefulPartitionedCall^HidB/StatefulPartitionedCall^HidB2/StatefulPartitionedCall^HidB3/StatefulPartitionedCall^HidC/StatefulPartitionedCall^HidC2/StatefulPartitionedCall^HidD1/StatefulPartitionedCall^HidE1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
ConA/StatefulPartitionedCallConA/StatefulPartitionedCall2>
ConA2/StatefulPartitionedCallConA2/StatefulPartitionedCall2>
ConA3/StatefulPartitionedCallConA3/StatefulPartitionedCall2>
ConA4/StatefulPartitionedCallConA4/StatefulPartitionedCall2<
ConB/StatefulPartitionedCallConB/StatefulPartitionedCall2>
ConB2/StatefulPartitionedCallConB2/StatefulPartitionedCall2>
ConB3/StatefulPartitionedCallConB3/StatefulPartitionedCall2<
ConC/StatefulPartitionedCallConC/StatefulPartitionedCall2>
ConC2/StatefulPartitionedCallConC2/StatefulPartitionedCall2<
ConD/StatefulPartitionedCallConD/StatefulPartitionedCall2<
ConE/StatefulPartitionedCallConE/StatefulPartitionedCall2<
Hid1/StatefulPartitionedCallHid1/StatefulPartitionedCall2<
Hid2/StatefulPartitionedCallHid2/StatefulPartitionedCall2<
HidA/StatefulPartitionedCallHidA/StatefulPartitionedCall2>
HidA2/StatefulPartitionedCallHidA2/StatefulPartitionedCall2>
HidA3/StatefulPartitionedCallHidA3/StatefulPartitionedCall2>
HidA4/StatefulPartitionedCallHidA4/StatefulPartitionedCall2<
HidB/StatefulPartitionedCallHidB/StatefulPartitionedCall2>
HidB2/StatefulPartitionedCallHidB2/StatefulPartitionedCall2>
HidB3/StatefulPartitionedCallHidB3/StatefulPartitionedCall2<
HidC/StatefulPartitionedCallHidC/StatefulPartitionedCall2>
HidC2/StatefulPartitionedCallHidC2/StatefulPartitionedCall2>
HidD1/StatefulPartitionedCallHidD1/StatefulPartitionedCall2>
HidE1/StatefulPartitionedCallHidE1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2F
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
?
C
'__inference_PoolA4_layer_call_fn_194545

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
B__inference_PoolA4_layer_call_and_return_conditional_losses_1909292
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
F
*__inference_flatten_5_layer_call_fn_194912

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
E__inference_flatten_5_layer_call_and_return_conditional_losses_1914132
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
?
?
%__inference_Hid2_layer_call_fn_195258

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
@__inference_Hid2_layer_call_and_return_conditional_losses_1916882
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
?
F
*__inference_flatten_8_layer_call_fn_194945

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
E__inference_flatten_8_layer_call_and_return_conditional_losses_1913892
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????":W S
/
_output_shapes
:?????????"
 
_user_specified_nameinputs
?
F
*__inference_dropout_8_layer_call_fn_194733

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
E__inference_dropout_8_layer_call_and_return_conditional_losses_1913442
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
?
?
A__inference_HidA4_layer_call_and_return_conditional_losses_191517

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
@__inference_Hid2_layer_call_and_return_conditional_losses_195249

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
?
c
*__inference_dropout_6_layer_call_fn_194684

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
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_1920802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_191323

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????G=2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????G=2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????G=:W S
/
_output_shapes
:?????????G=
 
_user_specified_nameinputs
?
]
A__inference_PoolD_layer_call_and_return_conditional_losses_191291

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????G=*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????G=2

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
a
(__inference_dropout_layer_call_fn_194176

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
:?????????I?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1924272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????I?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????I?22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????I?
 
_user_specified_nameinputs
?
c
*__inference_dropout_5_layer_call_fn_194657

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
E__inference_dropout_5_layer_call_and_return_conditional_losses_1920572
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
?
?
A__inference_ConA2_layer_call_and_return_conditional_losses_194207

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
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
:?????????I?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????I?
 
_user_specified_nameinputs
?
?
&__inference_HidD1_layer_call_fn_195167

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
GPU2*0J 8? *J
fERC
A__inference_HidD1_layer_call_and_return_conditional_losses_1916192
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
?
^
B__inference_PoolA3_layer_call_and_return_conditional_losses_194368

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
*__inference_dropout_2_layer_call_fn_194425

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
:?????????G>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_1911822
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????G>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????G>:W S
/
_output_shapes
:?????????G>
 
_user_specified_nameinputs
?
G
+__inference_flatten_10_layer_call_fn_194967

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
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_10_layer_call_and_return_conditional_losses_1913732
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????D; :W S
/
_output_shapes
:?????????D; 
 
_user_specified_nameinputs
?
F
*__inference_dropout_3_layer_call_fn_194814

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
:?????????G=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_1913232
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????G=2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????G=:W S
/
_output_shapes
:?????????G=
 
_user_specified_nameinputs
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_191129

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????H?2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????H?2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????H?:W S
/
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?
C
'__inference_PoolB2_layer_call_fn_194403

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
B__inference_PoolB2_layer_call_and_return_conditional_losses_1911882
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????F=:W S
/
_output_shapes
:?????????F=
 
_user_specified_nameinputs
?
?
A__inference_HidA3_layer_call_and_return_conditional_losses_191500

inputs2
matmul_readvariableop_resource:
?F?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?F?*
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
:??????????F: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????F
 
_user_specified_nameinputs
?
^
B__inference_PoolB2_layer_call_and_return_conditional_losses_190907

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
B__inference_PoolA2_layer_call_and_return_conditional_losses_194241

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
?
?
&__inference_ConA4_layer_call_fn_194450

inputs!
unknown:((
	unknown_0:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConA4_layer_call_and_return_conditional_losses_1912752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
^
B__inference_PoolC2_layer_call_and_return_conditional_losses_194580

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????"*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????"2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????E<:W S
/
_output_shapes
:?????????E<
 
_user_specified_nameinputs
?
d
E__inference_dropout_7_layer_call_and_return_conditional_losses_192103

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
:?????????(2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????(2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????(2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
]
A__inference_PoolD_layer_call_and_return_conditional_losses_194595

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
%__inference_ConE_layer_call_fn_194530

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConE_layer_call_and_return_conditional_losses_1912072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

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
?
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_194824

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????D; 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????D; 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????D; :W S
/
_output_shapes
:?????????D; 
 
_user_specified_nameinputs
?
?
&__inference_ConB3_layer_call_fn_194470

inputs!
unknown:(
	unknown_0:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConB3_layer_call_and_return_conditional_losses_1912582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????!(2

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
?
c
E__inference_dropout_6_layer_call_and_return_conditional_losses_191358

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????(2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????(2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
^
B__inference_PoolB3_layer_call_and_return_conditional_losses_194560

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????(*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????!(:W S
/
_output_shapes
:?????????!(
 
_user_specified_nameinputs
?
d
E__inference_dropout_8_layer_call_and_return_conditional_losses_194728

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
B__inference_output_layer_call_and_return_conditional_losses_191705

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
BiasAddb
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
Softmaxm
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:??????????2

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
?
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_191429

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_194635

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
?
?
%__inference_ConD_layer_call_fn_194510

inputs!
unknown:	
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
@__inference_ConD_layer_call_and_return_conditional_losses_1912242
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
?
]
A__inference_PoolD_layer_call_and_return_conditional_losses_190995

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
?
B
&__inference_PoolE_layer_call_fn_194630

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
:?????????D; * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolE_layer_call_and_return_conditional_losses_1912852
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????D; 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
@__inference_ConD_layer_call_and_return_conditional_losses_191224

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
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
?
d
E__inference_dropout_5_layer_call_and_return_conditional_losses_192057

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
^
B__inference_PoolC2_layer_call_and_return_conditional_losses_194575

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
&__inference_HidB2_layer_call_fn_195087

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
A__inference_HidB2_layer_call_and_return_conditional_losses_1915512
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
]
A__inference_PoolA_layer_call_and_return_conditional_losses_190797

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
?
]
A__inference_PoolB_layer_call_and_return_conditional_losses_194221

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
A__inference_ConB2_layer_call_and_return_conditional_losses_191154

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????F=*
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
:?????????F=2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????F=2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????F=2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????H?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?
c
E__inference_dropout_9_layer_call_and_return_conditional_losses_191337

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????(2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????(2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_194863

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
&__inference_PoolC_layer_call_fn_194363

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
:?????????G>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolC_layer_call_and_return_conditional_losses_1911412
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????G>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
A__inference_model_layer_call_and_return_conditional_losses_193096	
input%
cona_192936:
cona_192938:%
conb_192942:
conb_192944:&
cona2_192949:
cona2_192951:%
conc_192954:	
conc_192956:&
conb2_192962:
conb2_192964:&
cona3_192967:(
cona3_192969:(%
cone_192975: 
cone_192977: %
cond_192980:	
cond_192982:&
conc2_192985:
conc2_192987:&
conb3_192990:(
conb3_192992:(&
cona4_192995:((
cona4_192997:( 
hida_193024:???
hida_193026:	?!
hida2_193029:???
hida2_193031:	? 
hida3_193034:
?F?
hida3_193036:	? 
hida4_193039:
??
hida4_193041:	? 
hidb_193044:ȓ?
hidb_193046:	?!
hidb2_193049:???
hidb2_193051:	? 
hidb3_193054:
?F?
hidb3_193056:	? 
hidc_193059:???
hidc_193061:	?!
hidc2_193064:???
hidc2_193066:	?!
hidd1_193069:???
hidd1_193071:	?!
hide1_193074:???
hide1_193076:	?
hid1_193080:
??
hid1_193082:	?
hid2_193085:
??
hid2_193087:	?!
output_193090:
??
output_193092:	?
identity??ConA/StatefulPartitionedCall?ConA2/StatefulPartitionedCall?ConA3/StatefulPartitionedCall?ConA4/StatefulPartitionedCall?ConB/StatefulPartitionedCall?ConB2/StatefulPartitionedCall?ConB3/StatefulPartitionedCall?ConC/StatefulPartitionedCall?ConC2/StatefulPartitionedCall?ConD/StatefulPartitionedCall?ConE/StatefulPartitionedCall?Hid1/StatefulPartitionedCall?Hid2/StatefulPartitionedCall?HidA/StatefulPartitionedCall?HidA2/StatefulPartitionedCall?HidA3/StatefulPartitionedCall?HidA4/StatefulPartitionedCall?HidB/StatefulPartitionedCall?HidB2/StatefulPartitionedCall?HidB3/StatefulPartitionedCall?HidC/StatefulPartitionedCall?HidC2/StatefulPartitionedCall?HidD1/StatefulPartitionedCall?HidE1/StatefulPartitionedCall?output/StatefulPartitionedCall?
ConA/StatefulPartitionedCallStatefulPartitionedCallinputcona_192936cona_192938*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConA_layer_call_and_return_conditional_losses_1910482
ConA/StatefulPartitionedCall?
PoolA/PartitionedCallPartitionedCall%ConA/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolA_layer_call_and_return_conditional_losses_1910582
PoolA/PartitionedCall?
ConB/StatefulPartitionedCallStatefulPartitionedCallinputconb_192942conb_192944*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConB_layer_call_and_return_conditional_losses_1910712
ConB/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCallPoolA/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1910822
dropout/PartitionedCall?
PoolB/PartitionedCallPartitionedCall%ConB/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????H?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolB_layer_call_and_return_conditional_losses_1910882
PoolB/PartitionedCall?
ConA2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0cona2_192949cona2_192951*
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
A__inference_ConA2_layer_call_and_return_conditional_losses_1911012
ConA2/StatefulPartitionedCall?
ConC/StatefulPartitionedCallStatefulPartitionedCallinputconc_192954conc_192956*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConC_layer_call_and_return_conditional_losses_1911182
ConC/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCallPoolB/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????H?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1911292
dropout_1/PartitionedCall?
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
B__inference_PoolA2_layer_call_and_return_conditional_losses_1911352
PoolA2/PartitionedCall?
PoolC/PartitionedCallPartitionedCall%ConC/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????G>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolC_layer_call_and_return_conditional_losses_1911412
PoolC/PartitionedCall?
ConB2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conb2_192962conb2_192964*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConB2_layer_call_and_return_conditional_losses_1911542
ConB2/StatefulPartitionedCall?
ConA3/StatefulPartitionedCallStatefulPartitionedCallPoolA2/PartitionedCall:output:0cona3_192967cona3_192969*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConA3_layer_call_and_return_conditional_losses_1911712
ConA3/StatefulPartitionedCall?
dropout_2/PartitionedCallPartitionedCallPoolC/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????G>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_1911822
dropout_2/PartitionedCall?
PoolB2/PartitionedCallPartitionedCall&ConB2/StatefulPartitionedCall:output:0*
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
B__inference_PoolB2_layer_call_and_return_conditional_losses_1911882
PoolB2/PartitionedCall?
PoolA3/PartitionedCallPartitionedCall&ConA3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA3_layer_call_and_return_conditional_losses_1911942
PoolA3/PartitionedCall?
ConE/StatefulPartitionedCallStatefulPartitionedCallinputcone_192975cone_192977*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConE_layer_call_and_return_conditional_losses_1912072
ConE/StatefulPartitionedCall?
ConD/StatefulPartitionedCallStatefulPartitionedCallinputcond_192980cond_192982*
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
@__inference_ConD_layer_call_and_return_conditional_losses_1912242
ConD/StatefulPartitionedCall?
ConC2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0conc2_192985conc2_192987*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????E<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConC2_layer_call_and_return_conditional_losses_1912412
ConC2/StatefulPartitionedCall?
ConB3/StatefulPartitionedCallStatefulPartitionedCallPoolB2/PartitionedCall:output:0conb3_192990conb3_192992*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConB3_layer_call_and_return_conditional_losses_1912582
ConB3/StatefulPartitionedCall?
ConA4/StatefulPartitionedCallStatefulPartitionedCallPoolA3/PartitionedCall:output:0cona4_192995cona4_192997*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConA4_layer_call_and_return_conditional_losses_1912752
ConA4/StatefulPartitionedCall?
PoolE/PartitionedCallPartitionedCall%ConE/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????D; * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolE_layer_call_and_return_conditional_losses_1912852
PoolE/PartitionedCall?
PoolD/PartitionedCallPartitionedCall%ConD/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????G=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolD_layer_call_and_return_conditional_losses_1912912
PoolD/PartitionedCall?
PoolC2/PartitionedCallPartitionedCall&ConC2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolC2_layer_call_and_return_conditional_losses_1912972
PoolC2/PartitionedCall?
PoolB3/PartitionedCallPartitionedCall&ConB3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolB3_layer_call_and_return_conditional_losses_1913032
PoolB3/PartitionedCall?
PoolA4/PartitionedCallPartitionedCall&ConA4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA4_layer_call_and_return_conditional_losses_1913092
PoolA4/PartitionedCall?
dropout_4/PartitionedCallPartitionedCallPoolE/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????D; * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1913162
dropout_4/PartitionedCall?
dropout_3/PartitionedCallPartitionedCallPoolD/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????G=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_1913232
dropout_3/PartitionedCall?
dropout_10/PartitionedCallPartitionedCallPoolC2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_1913302
dropout_10/PartitionedCall?
dropout_9/PartitionedCallPartitionedCallPoolB3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_1913372
dropout_9/PartitionedCall?
dropout_8/PartitionedCallPartitionedCallPoolB2/PartitionedCall:output:0*
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
E__inference_dropout_8_layer_call_and_return_conditional_losses_1913442
dropout_8/PartitionedCall?
dropout_7/PartitionedCallPartitionedCallPoolA4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_1913512
dropout_7/PartitionedCall?
dropout_6/PartitionedCallPartitionedCallPoolA3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_1913582
dropout_6/PartitionedCall?
dropout_5/PartitionedCallPartitionedCallPoolA2/PartitionedCall:output:0*
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
E__inference_dropout_5_layer_call_and_return_conditional_losses_1913652
dropout_5/PartitionedCall?
flatten_10/PartitionedCallPartitionedCall"dropout_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_10_layer_call_and_return_conditional_losses_1913732
flatten_10/PartitionedCall?
flatten_9/PartitionedCallPartitionedCall"dropout_3/PartitionedCall:output:0*
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
GPU2*0J 8? *N
fIRG
E__inference_flatten_9_layer_call_and_return_conditional_losses_1913812
flatten_9/PartitionedCall?
flatten_8/PartitionedCallPartitionedCall#dropout_10/PartitionedCall:output:0*
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
E__inference_flatten_8_layer_call_and_return_conditional_losses_1913892
flatten_8/PartitionedCall?
flatten_7/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_1913972
flatten_7/PartitionedCall?
flatten_6/PartitionedCallPartitionedCall"dropout_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????F* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_1914052
flatten_6/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall"dropout_8/PartitionedCall:output:0*
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
E__inference_flatten_5_layer_call_and_return_conditional_losses_1914132
flatten_5/PartitionedCall?
flatten_4/PartitionedCallPartitionedCallPoolB/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????ȓ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_1914212
flatten_4/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall"dropout_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1914292
flatten_3/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall"dropout_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????F* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_1914372
flatten_2/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall"dropout_5/PartitionedCall:output:0*
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
E__inference_flatten_1_layer_call_and_return_conditional_losses_1914452
flatten_1/PartitionedCall?
flatten/PartitionedCallPartitionedCallPoolA/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1914532
flatten/PartitionedCall?
HidA/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0hida_193024hida_193026*
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
@__inference_HidA_layer_call_and_return_conditional_losses_1914662
HidA/StatefulPartitionedCall?
HidA2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0hida2_193029hida2_193031*
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
A__inference_HidA2_layer_call_and_return_conditional_losses_1914832
HidA2/StatefulPartitionedCall?
HidA3/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0hida3_193034hida3_193036*
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
A__inference_HidA3_layer_call_and_return_conditional_losses_1915002
HidA3/StatefulPartitionedCall?
HidA4/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0hida4_193039hida4_193041*
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
A__inference_HidA4_layer_call_and_return_conditional_losses_1915172
HidA4/StatefulPartitionedCall?
HidB/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0hidb_193044hidb_193046*
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
@__inference_HidB_layer_call_and_return_conditional_losses_1915342
HidB/StatefulPartitionedCall?
HidB2/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0hidb2_193049hidb2_193051*
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
A__inference_HidB2_layer_call_and_return_conditional_losses_1915512
HidB2/StatefulPartitionedCall?
HidB3/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0hidb3_193054hidb3_193056*
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
A__inference_HidB3_layer_call_and_return_conditional_losses_1915682
HidB3/StatefulPartitionedCall?
HidC/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0hidc_193059hidc_193061*
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
@__inference_HidC_layer_call_and_return_conditional_losses_1915852
HidC/StatefulPartitionedCall?
HidC2/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0hidc2_193064hidc2_193066*
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
A__inference_HidC2_layer_call_and_return_conditional_losses_1916022
HidC2/StatefulPartitionedCall?
HidD1/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0hidd1_193069hidd1_193071*
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
A__inference_HidD1_layer_call_and_return_conditional_losses_1916192
HidD1/StatefulPartitionedCall?
HidE1/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0hide1_193074hide1_193076*
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
A__inference_HidE1_layer_call_and_return_conditional_losses_1916362
HidE1/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall%HidA/StatefulPartitionedCall:output:0&HidA2/StatefulPartitionedCall:output:0&HidA3/StatefulPartitionedCall:output:0&HidA4/StatefulPartitionedCall:output:0%HidB/StatefulPartitionedCall:output:0&HidB2/StatefulPartitionedCall:output:0&HidB3/StatefulPartitionedCall:output:0%HidC/StatefulPartitionedCall:output:0&HidC2/StatefulPartitionedCall:output:0&HidD1/StatefulPartitionedCall:output:0&HidE1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_1916582
concatenate/PartitionedCall?
Hid1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0hid1_193080hid1_193082*
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
@__inference_Hid1_layer_call_and_return_conditional_losses_1916712
Hid1/StatefulPartitionedCall?
Hid2/StatefulPartitionedCallStatefulPartitionedCall%Hid1/StatefulPartitionedCall:output:0hid2_193085hid2_193087*
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
@__inference_Hid2_layer_call_and_return_conditional_losses_1916882
Hid2/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall%Hid2/StatefulPartitionedCall:output:0output_193090output_193092*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1917052 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^ConA/StatefulPartitionedCall^ConA2/StatefulPartitionedCall^ConA3/StatefulPartitionedCall^ConA4/StatefulPartitionedCall^ConB/StatefulPartitionedCall^ConB2/StatefulPartitionedCall^ConB3/StatefulPartitionedCall^ConC/StatefulPartitionedCall^ConC2/StatefulPartitionedCall^ConD/StatefulPartitionedCall^ConE/StatefulPartitionedCall^Hid1/StatefulPartitionedCall^Hid2/StatefulPartitionedCall^HidA/StatefulPartitionedCall^HidA2/StatefulPartitionedCall^HidA3/StatefulPartitionedCall^HidA4/StatefulPartitionedCall^HidB/StatefulPartitionedCall^HidB2/StatefulPartitionedCall^HidB3/StatefulPartitionedCall^HidC/StatefulPartitionedCall^HidC2/StatefulPartitionedCall^HidD1/StatefulPartitionedCall^HidE1/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
ConA/StatefulPartitionedCallConA/StatefulPartitionedCall2>
ConA2/StatefulPartitionedCallConA2/StatefulPartitionedCall2>
ConA3/StatefulPartitionedCallConA3/StatefulPartitionedCall2>
ConA4/StatefulPartitionedCallConA4/StatefulPartitionedCall2<
ConB/StatefulPartitionedCallConB/StatefulPartitionedCall2>
ConB2/StatefulPartitionedCallConB2/StatefulPartitionedCall2>
ConB3/StatefulPartitionedCallConB3/StatefulPartitionedCall2<
ConC/StatefulPartitionedCallConC/StatefulPartitionedCall2>
ConC2/StatefulPartitionedCallConC2/StatefulPartitionedCall2<
ConD/StatefulPartitionedCallConD/StatefulPartitionedCall2<
ConE/StatefulPartitionedCallConE/StatefulPartitionedCall2<
Hid1/StatefulPartitionedCallHid1/StatefulPartitionedCall2<
Hid2/StatefulPartitionedCallHid2/StatefulPartitionedCall2<
HidA/StatefulPartitionedCallHidA/StatefulPartitionedCall2>
HidA2/StatefulPartitionedCallHidA2/StatefulPartitionedCall2>
HidA3/StatefulPartitionedCallHidA3/StatefulPartitionedCall2>
HidA4/StatefulPartitionedCallHidA4/StatefulPartitionedCall2<
HidB/StatefulPartitionedCallHidB/StatefulPartitionedCall2>
HidB2/StatefulPartitionedCallHidB2/StatefulPartitionedCall2>
HidB3/StatefulPartitionedCallHidB3/StatefulPartitionedCall2<
HidC/StatefulPartitionedCallHidC/StatefulPartitionedCall2>
HidC2/StatefulPartitionedCallHidC2/StatefulPartitionedCall2>
HidD1/StatefulPartitionedCallHidD1/StatefulPartitionedCall2>
HidE1/StatefulPartitionedCallHidE1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
?
A__inference_ConB3_layer_call_and_return_conditional_losses_194461

inputs8
conv2d_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!(*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!(2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????!(2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????!(2

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
?
B
&__inference_PoolD_layer_call_fn_194610

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
:?????????G=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolD_layer_call_and_return_conditional_losses_1912912
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????G=2

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
A__inference_HidA2_layer_call_and_return_conditional_losses_194998

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
^
B__inference_PoolC2_layer_call_and_return_conditional_losses_191297

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????"*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????"2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????E<:W S
/
_output_shapes
:?????????E<
 
_user_specified_nameinputs
?
?
A__inference_HidE1_layer_call_and_return_conditional_losses_195178

inputs3
matmul_readvariableop_resource:???.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
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
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
@__inference_ConE_layer_call_and_return_conditional_losses_191207

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:??????????? 2

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
?
?
&__inference_model_layer_call_fn_192933	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:(

unknown_10:($

unknown_11: 

unknown_12: $

unknown_13:	

unknown_14:$

unknown_15:

unknown_16:$

unknown_17:(

unknown_18:($

unknown_19:((

unknown_20:(

unknown_21:???

unknown_22:	?

unknown_23:???

unknown_24:	?

unknown_25:
?F?

unknown_26:	?

unknown_27:
??

unknown_28:	?

unknown_29:ȓ?

unknown_30:	?

unknown_31:???

unknown_32:	?

unknown_33:
?F?

unknown_34:	?

unknown_35:???

unknown_36:	?

unknown_37:???

unknown_38:	?

unknown_39:???

unknown_40:	?

unknown_41:???

unknown_42:	?

unknown_43:
??

unknown_44:	?

unknown_45:
??

unknown_46:	?

unknown_47:
??

unknown_48:	?
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
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1927252
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
a
E__inference_flatten_6_layer_call_and_return_conditional_losses_194918

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? #  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????F2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????F2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
A__inference_ConA4_layer_call_and_return_conditional_losses_191275

inputs8
conv2d_readvariableop_resource:((-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:((*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????(2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
c
*__inference_dropout_7_layer_call_fn_194711

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
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_1921032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
B
&__inference_PoolA_layer_call_fn_194149

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
:?????????I?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolA_layer_call_and_return_conditional_losses_1910582
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????I?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
^
B__inference_PoolA2_layer_call_and_return_conditional_losses_190841

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
C
'__inference_PoolB3_layer_call_fn_194570

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
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolB3_layer_call_and_return_conditional_losses_1913032
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????!(:W S
/
_output_shapes
:?????????!(
 
_user_specified_nameinputs
?
?
A__inference_ConA2_layer_call_and_return_conditional_losses_191101

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
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
:?????????I?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????I?
 
_user_specified_nameinputs
?
]
A__inference_PoolB_layer_call_and_return_conditional_losses_191088

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????H?*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????H?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
@__inference_HidB_layer_call_and_return_conditional_losses_195058

inputs3
matmul_readvariableop_resource:ȓ?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:ȓ?*
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
:?????????ȓ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:?????????ȓ
 
_user_specified_nameinputs
?
?
A__inference_HidA3_layer_call_and_return_conditional_losses_195018

inputs2
matmul_readvariableop_resource:
?F?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?F?*
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
:??????????F: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????F
 
_user_specified_nameinputs
?
a
E__inference_flatten_8_layer_call_and_return_conditional_losses_194940

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????O  2
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
:?????????":W S
/
_output_shapes
:?????????"
 
_user_specified_nameinputs
??
?a
"__inference__traced_restore_196265
file_prefix6
assignvariableop_cona_kernel:*
assignvariableop_1_cona_bias:8
assignvariableop_2_conb_kernel:*
assignvariableop_3_conb_bias:9
assignvariableop_4_cona2_kernel:+
assignvariableop_5_cona2_bias:8
assignvariableop_6_conc_kernel:	*
assignvariableop_7_conc_bias:9
assignvariableop_8_cona3_kernel:(+
assignvariableop_9_cona3_bias:(:
 assignvariableop_10_conb2_kernel:,
assignvariableop_11_conb2_bias::
 assignvariableop_12_cona4_kernel:((,
assignvariableop_13_cona4_bias:(:
 assignvariableop_14_conb3_kernel:(,
assignvariableop_15_conb3_bias:(:
 assignvariableop_16_conc2_kernel:,
assignvariableop_17_conc2_bias:9
assignvariableop_18_cond_kernel:	+
assignvariableop_19_cond_bias:9
assignvariableop_20_cone_kernel: +
assignvariableop_21_cone_bias: 4
assignvariableop_22_hida_kernel:???,
assignvariableop_23_hida_bias:	?5
 assignvariableop_24_hida2_kernel:???-
assignvariableop_25_hida2_bias:	?4
 assignvariableop_26_hida3_kernel:
?F?-
assignvariableop_27_hida3_bias:	?4
 assignvariableop_28_hida4_kernel:
??-
assignvariableop_29_hida4_bias:	?4
assignvariableop_30_hidb_kernel:ȓ?,
assignvariableop_31_hidb_bias:	?5
 assignvariableop_32_hidb2_kernel:???-
assignvariableop_33_hidb2_bias:	?4
 assignvariableop_34_hidb3_kernel:
?F?-
assignvariableop_35_hidb3_bias:	?4
assignvariableop_36_hidc_kernel:???,
assignvariableop_37_hidc_bias:	?5
 assignvariableop_38_hidc2_kernel:???-
assignvariableop_39_hidc2_bias:	?5
 assignvariableop_40_hidd1_kernel:???-
assignvariableop_41_hidd1_bias:	?5
 assignvariableop_42_hide1_kernel:???-
assignvariableop_43_hide1_bias:	?3
assignvariableop_44_hid1_kernel:
??,
assignvariableop_45_hid1_bias:	?3
assignvariableop_46_hid2_kernel:
??,
assignvariableop_47_hid2_bias:	?5
!assignvariableop_48_output_kernel:
??.
assignvariableop_49_output_bias:	?'
assignvariableop_50_adam_iter:	 )
assignvariableop_51_adam_beta_1: )
assignvariableop_52_adam_beta_2: (
assignvariableop_53_adam_decay: 0
&assignvariableop_54_adam_learning_rate: #
assignvariableop_55_total: #
assignvariableop_56_count: %
assignvariableop_57_total_1: %
assignvariableop_58_count_1: @
&assignvariableop_59_adam_cona_kernel_m:2
$assignvariableop_60_adam_cona_bias_m:@
&assignvariableop_61_adam_conb_kernel_m:2
$assignvariableop_62_adam_conb_bias_m:A
'assignvariableop_63_adam_cona2_kernel_m:3
%assignvariableop_64_adam_cona2_bias_m:@
&assignvariableop_65_adam_conc_kernel_m:	2
$assignvariableop_66_adam_conc_bias_m:A
'assignvariableop_67_adam_cona3_kernel_m:(3
%assignvariableop_68_adam_cona3_bias_m:(A
'assignvariableop_69_adam_conb2_kernel_m:3
%assignvariableop_70_adam_conb2_bias_m:A
'assignvariableop_71_adam_cona4_kernel_m:((3
%assignvariableop_72_adam_cona4_bias_m:(A
'assignvariableop_73_adam_conb3_kernel_m:(3
%assignvariableop_74_adam_conb3_bias_m:(A
'assignvariableop_75_adam_conc2_kernel_m:3
%assignvariableop_76_adam_conc2_bias_m:@
&assignvariableop_77_adam_cond_kernel_m:	2
$assignvariableop_78_adam_cond_bias_m:@
&assignvariableop_79_adam_cone_kernel_m: 2
$assignvariableop_80_adam_cone_bias_m: ;
&assignvariableop_81_adam_hida_kernel_m:???3
$assignvariableop_82_adam_hida_bias_m:	?<
'assignvariableop_83_adam_hida2_kernel_m:???4
%assignvariableop_84_adam_hida2_bias_m:	?;
'assignvariableop_85_adam_hida3_kernel_m:
?F?4
%assignvariableop_86_adam_hida3_bias_m:	?;
'assignvariableop_87_adam_hida4_kernel_m:
??4
%assignvariableop_88_adam_hida4_bias_m:	?;
&assignvariableop_89_adam_hidb_kernel_m:ȓ?3
$assignvariableop_90_adam_hidb_bias_m:	?<
'assignvariableop_91_adam_hidb2_kernel_m:???4
%assignvariableop_92_adam_hidb2_bias_m:	?;
'assignvariableop_93_adam_hidb3_kernel_m:
?F?4
%assignvariableop_94_adam_hidb3_bias_m:	?;
&assignvariableop_95_adam_hidc_kernel_m:???3
$assignvariableop_96_adam_hidc_bias_m:	?<
'assignvariableop_97_adam_hidc2_kernel_m:???4
%assignvariableop_98_adam_hidc2_bias_m:	?<
'assignvariableop_99_adam_hidd1_kernel_m:???5
&assignvariableop_100_adam_hidd1_bias_m:	?=
(assignvariableop_101_adam_hide1_kernel_m:???5
&assignvariableop_102_adam_hide1_bias_m:	?;
'assignvariableop_103_adam_hid1_kernel_m:
??4
%assignvariableop_104_adam_hid1_bias_m:	?;
'assignvariableop_105_adam_hid2_kernel_m:
??4
%assignvariableop_106_adam_hid2_bias_m:	?=
)assignvariableop_107_adam_output_kernel_m:
??6
'assignvariableop_108_adam_output_bias_m:	?A
'assignvariableop_109_adam_cona_kernel_v:3
%assignvariableop_110_adam_cona_bias_v:A
'assignvariableop_111_adam_conb_kernel_v:3
%assignvariableop_112_adam_conb_bias_v:B
(assignvariableop_113_adam_cona2_kernel_v:4
&assignvariableop_114_adam_cona2_bias_v:A
'assignvariableop_115_adam_conc_kernel_v:	3
%assignvariableop_116_adam_conc_bias_v:B
(assignvariableop_117_adam_cona3_kernel_v:(4
&assignvariableop_118_adam_cona3_bias_v:(B
(assignvariableop_119_adam_conb2_kernel_v:4
&assignvariableop_120_adam_conb2_bias_v:B
(assignvariableop_121_adam_cona4_kernel_v:((4
&assignvariableop_122_adam_cona4_bias_v:(B
(assignvariableop_123_adam_conb3_kernel_v:(4
&assignvariableop_124_adam_conb3_bias_v:(B
(assignvariableop_125_adam_conc2_kernel_v:4
&assignvariableop_126_adam_conc2_bias_v:A
'assignvariableop_127_adam_cond_kernel_v:	3
%assignvariableop_128_adam_cond_bias_v:A
'assignvariableop_129_adam_cone_kernel_v: 3
%assignvariableop_130_adam_cone_bias_v: <
'assignvariableop_131_adam_hida_kernel_v:???4
%assignvariableop_132_adam_hida_bias_v:	?=
(assignvariableop_133_adam_hida2_kernel_v:???5
&assignvariableop_134_adam_hida2_bias_v:	?<
(assignvariableop_135_adam_hida3_kernel_v:
?F?5
&assignvariableop_136_adam_hida3_bias_v:	?<
(assignvariableop_137_adam_hida4_kernel_v:
??5
&assignvariableop_138_adam_hida4_bias_v:	?<
'assignvariableop_139_adam_hidb_kernel_v:ȓ?4
%assignvariableop_140_adam_hidb_bias_v:	?=
(assignvariableop_141_adam_hidb2_kernel_v:???5
&assignvariableop_142_adam_hidb2_bias_v:	?<
(assignvariableop_143_adam_hidb3_kernel_v:
?F?5
&assignvariableop_144_adam_hidb3_bias_v:	?<
'assignvariableop_145_adam_hidc_kernel_v:???4
%assignvariableop_146_adam_hidc_bias_v:	?=
(assignvariableop_147_adam_hidc2_kernel_v:???5
&assignvariableop_148_adam_hidc2_bias_v:	?=
(assignvariableop_149_adam_hidd1_kernel_v:???5
&assignvariableop_150_adam_hidd1_bias_v:	?=
(assignvariableop_151_adam_hide1_kernel_v:???5
&assignvariableop_152_adam_hide1_bias_v:	?;
'assignvariableop_153_adam_hid1_kernel_v:
??4
%assignvariableop_154_adam_hid1_bias_v:	?;
'assignvariableop_155_adam_hid2_kernel_v:
??4
%assignvariableop_156_adam_hid2_bias_v:	?=
)assignvariableop_157_adam_output_kernel_v:
??6
'assignvariableop_158_adam_output_bias_v:	?
identity_160??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_120?AssignVariableOp_121?AssignVariableOp_122?AssignVariableOp_123?AssignVariableOp_124?AssignVariableOp_125?AssignVariableOp_126?AssignVariableOp_127?AssignVariableOp_128?AssignVariableOp_129?AssignVariableOp_13?AssignVariableOp_130?AssignVariableOp_131?AssignVariableOp_132?AssignVariableOp_133?AssignVariableOp_134?AssignVariableOp_135?AssignVariableOp_136?AssignVariableOp_137?AssignVariableOp_138?AssignVariableOp_139?AssignVariableOp_14?AssignVariableOp_140?AssignVariableOp_141?AssignVariableOp_142?AssignVariableOp_143?AssignVariableOp_144?AssignVariableOp_145?AssignVariableOp_146?AssignVariableOp_147?AssignVariableOp_148?AssignVariableOp_149?AssignVariableOp_15?AssignVariableOp_150?AssignVariableOp_151?AssignVariableOp_152?AssignVariableOp_153?AssignVariableOp_154?AssignVariableOp_155?AssignVariableOp_156?AssignVariableOp_157?AssignVariableOp_158?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?[
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?Z
value?ZB?Z?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
?2?	2
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
AssignVariableOp_2AssignVariableOpassignvariableop_2_conb_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_conb_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_cona2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_cona2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_conc_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_conc_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_cona3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_cona3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp assignvariableop_10_conb2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_conb2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp assignvariableop_12_cona4_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_cona4_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp assignvariableop_14_conb3_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_conb3_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp assignvariableop_16_conc2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_conc2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_cond_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_cond_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_cone_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_cone_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_hida_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_hida_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp assignvariableop_24_hida2_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_hida2_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp assignvariableop_26_hida3_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_hida3_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp assignvariableop_28_hida4_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_hida4_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_hidb_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_hidb_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp assignvariableop_32_hidb2_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_hidb2_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp assignvariableop_34_hidb3_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpassignvariableop_35_hidb3_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpassignvariableop_36_hidc_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpassignvariableop_37_hidc_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp assignvariableop_38_hidc2_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpassignvariableop_39_hidc2_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp assignvariableop_40_hidd1_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpassignvariableop_41_hidd1_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp assignvariableop_42_hide1_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpassignvariableop_43_hide1_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpassignvariableop_44_hid1_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpassignvariableop_45_hid1_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOpassignvariableop_46_hid2_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOpassignvariableop_47_hid2_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp!assignvariableop_48_output_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOpassignvariableop_49_output_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpassignvariableop_50_adam_iterIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOpassignvariableop_51_adam_beta_1Identity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOpassignvariableop_52_adam_beta_2Identity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOpassignvariableop_53_adam_decayIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp&assignvariableop_54_adam_learning_rateIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOpassignvariableop_55_totalIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOpassignvariableop_56_countIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOpassignvariableop_57_total_1Identity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOpassignvariableop_58_count_1Identity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp&assignvariableop_59_adam_cona_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp$assignvariableop_60_adam_cona_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp&assignvariableop_61_adam_conb_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp$assignvariableop_62_adam_conb_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp'assignvariableop_63_adam_cona2_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp%assignvariableop_64_adam_cona2_bias_mIdentity_64:output:0"/device:CPU:0*
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
AssignVariableOp_67AssignVariableOp'assignvariableop_67_adam_cona3_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp%assignvariableop_68_adam_cona3_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp'assignvariableop_69_adam_conb2_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp%assignvariableop_70_adam_conb2_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp'assignvariableop_71_adam_cona4_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp%assignvariableop_72_adam_cona4_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp'assignvariableop_73_adam_conb3_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp%assignvariableop_74_adam_conb3_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp'assignvariableop_75_adam_conc2_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp%assignvariableop_76_adam_conc2_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp&assignvariableop_77_adam_cond_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp$assignvariableop_78_adam_cond_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp&assignvariableop_79_adam_cone_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp$assignvariableop_80_adam_cone_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp&assignvariableop_81_adam_hida_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp$assignvariableop_82_adam_hida_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp'assignvariableop_83_adam_hida2_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp%assignvariableop_84_adam_hida2_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp'assignvariableop_85_adam_hida3_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp%assignvariableop_86_adam_hida3_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp'assignvariableop_87_adam_hida4_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp%assignvariableop_88_adam_hida4_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp&assignvariableop_89_adam_hidb_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp$assignvariableop_90_adam_hidb_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp'assignvariableop_91_adam_hidb2_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp%assignvariableop_92_adam_hidb2_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp'assignvariableop_93_adam_hidb3_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp%assignvariableop_94_adam_hidb3_bias_mIdentity_94:output:0"/device:CPU:0*
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
AssignVariableOp_99AssignVariableOp'assignvariableop_99_adam_hidd1_kernel_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOp&assignvariableop_100_adam_hidd1_bias_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOp(assignvariableop_101_adam_hide1_kernel_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOp&assignvariableop_102_adam_hide1_bias_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOp'assignvariableop_103_adam_hid1_kernel_mIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104?
AssignVariableOp_104AssignVariableOp%assignvariableop_104_adam_hid1_bias_mIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105?
AssignVariableOp_105AssignVariableOp'assignvariableop_105_adam_hid2_kernel_mIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106?
AssignVariableOp_106AssignVariableOp%assignvariableop_106_adam_hid2_bias_mIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107?
AssignVariableOp_107AssignVariableOp)assignvariableop_107_adam_output_kernel_mIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108?
AssignVariableOp_108AssignVariableOp'assignvariableop_108_adam_output_bias_mIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109?
AssignVariableOp_109AssignVariableOp'assignvariableop_109_adam_cona_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110?
AssignVariableOp_110AssignVariableOp%assignvariableop_110_adam_cona_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111?
AssignVariableOp_111AssignVariableOp'assignvariableop_111_adam_conb_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112?
AssignVariableOp_112AssignVariableOp%assignvariableop_112_adam_conb_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113?
AssignVariableOp_113AssignVariableOp(assignvariableop_113_adam_cona2_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114?
AssignVariableOp_114AssignVariableOp&assignvariableop_114_adam_cona2_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115?
AssignVariableOp_115AssignVariableOp'assignvariableop_115_adam_conc_kernel_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116?
AssignVariableOp_116AssignVariableOp%assignvariableop_116_adam_conc_bias_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117?
AssignVariableOp_117AssignVariableOp(assignvariableop_117_adam_cona3_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118?
AssignVariableOp_118AssignVariableOp&assignvariableop_118_adam_cona3_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119?
AssignVariableOp_119AssignVariableOp(assignvariableop_119_adam_conb2_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120?
AssignVariableOp_120AssignVariableOp&assignvariableop_120_adam_conb2_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121?
AssignVariableOp_121AssignVariableOp(assignvariableop_121_adam_cona4_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122?
AssignVariableOp_122AssignVariableOp&assignvariableop_122_adam_cona4_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123?
AssignVariableOp_123AssignVariableOp(assignvariableop_123_adam_conb3_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124?
AssignVariableOp_124AssignVariableOp&assignvariableop_124_adam_conb3_bias_vIdentity_124:output:0"/device:CPU:0*
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
AssignVariableOp_127AssignVariableOp'assignvariableop_127_adam_cond_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128?
AssignVariableOp_128AssignVariableOp%assignvariableop_128_adam_cond_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129?
AssignVariableOp_129AssignVariableOp'assignvariableop_129_adam_cone_kernel_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130?
AssignVariableOp_130AssignVariableOp%assignvariableop_130_adam_cone_bias_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131?
AssignVariableOp_131AssignVariableOp'assignvariableop_131_adam_hida_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132?
AssignVariableOp_132AssignVariableOp%assignvariableop_132_adam_hida_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_132q
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:2
Identity_133?
AssignVariableOp_133AssignVariableOp(assignvariableop_133_adam_hida2_kernel_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_133q
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:2
Identity_134?
AssignVariableOp_134AssignVariableOp&assignvariableop_134_adam_hida2_bias_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_134q
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:2
Identity_135?
AssignVariableOp_135AssignVariableOp(assignvariableop_135_adam_hida3_kernel_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_135q
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:2
Identity_136?
AssignVariableOp_136AssignVariableOp&assignvariableop_136_adam_hida3_bias_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_136q
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:2
Identity_137?
AssignVariableOp_137AssignVariableOp(assignvariableop_137_adam_hida4_kernel_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_137q
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:2
Identity_138?
AssignVariableOp_138AssignVariableOp&assignvariableop_138_adam_hida4_bias_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_138q
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:2
Identity_139?
AssignVariableOp_139AssignVariableOp'assignvariableop_139_adam_hidb_kernel_vIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139q
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:2
Identity_140?
AssignVariableOp_140AssignVariableOp%assignvariableop_140_adam_hidb_bias_vIdentity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_140q
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:2
Identity_141?
AssignVariableOp_141AssignVariableOp(assignvariableop_141_adam_hidb2_kernel_vIdentity_141:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_141q
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:2
Identity_142?
AssignVariableOp_142AssignVariableOp&assignvariableop_142_adam_hidb2_bias_vIdentity_142:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_142q
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:2
Identity_143?
AssignVariableOp_143AssignVariableOp(assignvariableop_143_adam_hidb3_kernel_vIdentity_143:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_143q
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:2
Identity_144?
AssignVariableOp_144AssignVariableOp&assignvariableop_144_adam_hidb3_bias_vIdentity_144:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_144q
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
:2
Identity_145?
AssignVariableOp_145AssignVariableOp'assignvariableop_145_adam_hidc_kernel_vIdentity_145:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_145q
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:2
Identity_146?
AssignVariableOp_146AssignVariableOp%assignvariableop_146_adam_hidc_bias_vIdentity_146:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_146q
Identity_147IdentityRestoreV2:tensors:147"/device:CPU:0*
T0*
_output_shapes
:2
Identity_147?
AssignVariableOp_147AssignVariableOp(assignvariableop_147_adam_hidc2_kernel_vIdentity_147:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_147q
Identity_148IdentityRestoreV2:tensors:148"/device:CPU:0*
T0*
_output_shapes
:2
Identity_148?
AssignVariableOp_148AssignVariableOp&assignvariableop_148_adam_hidc2_bias_vIdentity_148:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_148q
Identity_149IdentityRestoreV2:tensors:149"/device:CPU:0*
T0*
_output_shapes
:2
Identity_149?
AssignVariableOp_149AssignVariableOp(assignvariableop_149_adam_hidd1_kernel_vIdentity_149:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_149q
Identity_150IdentityRestoreV2:tensors:150"/device:CPU:0*
T0*
_output_shapes
:2
Identity_150?
AssignVariableOp_150AssignVariableOp&assignvariableop_150_adam_hidd1_bias_vIdentity_150:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_150q
Identity_151IdentityRestoreV2:tensors:151"/device:CPU:0*
T0*
_output_shapes
:2
Identity_151?
AssignVariableOp_151AssignVariableOp(assignvariableop_151_adam_hide1_kernel_vIdentity_151:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_151q
Identity_152IdentityRestoreV2:tensors:152"/device:CPU:0*
T0*
_output_shapes
:2
Identity_152?
AssignVariableOp_152AssignVariableOp&assignvariableop_152_adam_hide1_bias_vIdentity_152:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_152q
Identity_153IdentityRestoreV2:tensors:153"/device:CPU:0*
T0*
_output_shapes
:2
Identity_153?
AssignVariableOp_153AssignVariableOp'assignvariableop_153_adam_hid1_kernel_vIdentity_153:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_153q
Identity_154IdentityRestoreV2:tensors:154"/device:CPU:0*
T0*
_output_shapes
:2
Identity_154?
AssignVariableOp_154AssignVariableOp%assignvariableop_154_adam_hid1_bias_vIdentity_154:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_154q
Identity_155IdentityRestoreV2:tensors:155"/device:CPU:0*
T0*
_output_shapes
:2
Identity_155?
AssignVariableOp_155AssignVariableOp'assignvariableop_155_adam_hid2_kernel_vIdentity_155:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_155q
Identity_156IdentityRestoreV2:tensors:156"/device:CPU:0*
T0*
_output_shapes
:2
Identity_156?
AssignVariableOp_156AssignVariableOp%assignvariableop_156_adam_hid2_bias_vIdentity_156:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_156q
Identity_157IdentityRestoreV2:tensors:157"/device:CPU:0*
T0*
_output_shapes
:2
Identity_157?
AssignVariableOp_157AssignVariableOp)assignvariableop_157_adam_output_kernel_vIdentity_157:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_157q
Identity_158IdentityRestoreV2:tensors:158"/device:CPU:0*
T0*
_output_shapes
:2
Identity_158?
AssignVariableOp_158AssignVariableOp'assignvariableop_158_adam_output_bias_vIdentity_158:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1589
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_159Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_159i
Identity_160IdentityIdentity_159:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_160?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"%
identity_160Identity_160:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_158AssignVariableOp_1582*
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
?
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_191182

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????G>2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????G>2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????G>:W S
/
_output_shapes
:?????????G>
 
_user_specified_nameinputs
?
]
A__inference_PoolA_layer_call_and_return_conditional_losses_194139

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????I?*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????I?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
C__inference_dropout_layer_call_and_return_conditional_losses_192427

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
:?????????I?2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????I?*
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
:?????????I?2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????I?2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????I?2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????I?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????I?:W S
/
_output_shapes
:?????????I?
 
_user_specified_nameinputs
?
?
@__inference_Hid1_layer_call_and_return_conditional_losses_191671

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
]
A__inference_PoolE_layer_call_and_return_conditional_losses_191017

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
?
%__inference_Hid1_layer_call_fn_195238

inputs
unknown:
??
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
@__inference_Hid1_layer_call_and_return_conditional_losses_1916712
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
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
@__inference_Hid2_layer_call_and_return_conditional_losses_191688

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
A__inference_ConA3_layer_call_and_return_conditional_losses_194314

inputs8
conv2d_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!(*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!(2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????!(2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????!(2

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
]
A__inference_PoolB_layer_call_and_return_conditional_losses_194226

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????H?*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????H?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_194809

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
:?????????G=2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????G=*
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
:?????????G=2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????G=2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????G=2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????G=2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????G=:W S
/
_output_shapes
:?????????G=
 
_user_specified_nameinputs
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_194852

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"??????  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????I?:W S
/
_output_shapes
:?????????I?
 
_user_specified_nameinputs
?
c
*__inference_dropout_3_layer_call_fn_194819

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
:?????????G=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_1921952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????G=2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????G=22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????G=
 
_user_specified_nameinputs
?
a
E__inference_flatten_7_layer_call_and_return_conditional_losses_194929

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????W 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????G>:W S
/
_output_shapes
:?????????G>
 
_user_specified_nameinputs
?
C
'__inference_PoolB2_layer_call_fn_194398

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
B__inference_PoolB2_layer_call_and_return_conditional_losses_1909072
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
@__inference_HidA_layer_call_and_return_conditional_losses_194978

inputs3
matmul_readvariableop_resource:???.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
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
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
C
'__inference_PoolA3_layer_call_fn_194378

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
B__inference_PoolA3_layer_call_and_return_conditional_losses_1908852
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
E__inference_flatten_2_layer_call_and_return_conditional_losses_194874

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? #  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????F2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????F2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
^
B__inference_PoolA3_layer_call_and_return_conditional_losses_191194

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????(*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????!(:W S
/
_output_shapes
:?????????!(
 
_user_specified_nameinputs
?
C
'__inference_PoolA2_layer_call_fn_194251

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
B__inference_PoolA2_layer_call_and_return_conditional_losses_1908412
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
?
?
$__inference_signature_wrapper_193372	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:(

unknown_10:($

unknown_11: 

unknown_12: $

unknown_13:	

unknown_14:$

unknown_15:

unknown_16:$

unknown_17:(

unknown_18:($

unknown_19:((

unknown_20:(

unknown_21:???

unknown_22:	?

unknown_23:???

unknown_24:	?

unknown_25:
?F?

unknown_26:	?

unknown_27:
??

unknown_28:	?

unknown_29:ȓ?

unknown_30:	?

unknown_31:???

unknown_32:	?

unknown_33:
?F?

unknown_34:	?

unknown_35:???

unknown_36:	?

unknown_37:???

unknown_38:	?

unknown_39:???

unknown_40:	?

unknown_41:???

unknown_42:	?

unknown_43:
??

unknown_44:	?

unknown_45:
??

unknown_46:	?

unknown_47:
??

unknown_48:	?
identity??StatefulPartitionedCall?
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
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_1907882
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
?
@__inference_ConA_layer_call_and_return_conditional_losses_191048

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????2

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
*__inference_dropout_4_layer_call_fn_194841

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
:?????????D; * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1913162
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????D; 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????D; :W S
/
_output_shapes
:?????????D; 
 
_user_specified_nameinputs
?
G
+__inference_dropout_10_layer_call_fn_194787

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
:?????????"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_1913302
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????":W S
/
_output_shapes
:?????????"
 
_user_specified_nameinputs
?
?
B__inference_output_layer_call_and_return_conditional_losses_195269

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
BiasAddb
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
Softmaxm
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:??????????2

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
?
c
*__inference_dropout_1_layer_call_fn_194283

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
:?????????H?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1923792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????H?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????H?22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?
c
*__inference_dropout_4_layer_call_fn_194846

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
:?????????D; * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1922182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????D; 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????D; 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????D; 
 
_user_specified_nameinputs
?
?
%__inference_ConB_layer_call_fn_194196

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConB_layer_call_and_return_conditional_losses_1910712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

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
?
a
E__inference_flatten_5_layer_call_and_return_conditional_losses_194907

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
c
*__inference_dropout_8_layer_call_fn_194738

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
E__inference_dropout_8_layer_call_and_return_conditional_losses_1921262
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
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_194261

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????H?2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????H?2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????H?:W S
/
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?
^
B__inference_PoolB2_layer_call_and_return_conditional_losses_194388

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
e
F__inference_dropout_10_layer_call_and_return_conditional_losses_194782

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
:?????????"2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????"*
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
:?????????"2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????"2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????"2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????"2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????":W S
/
_output_shapes
:?????????"
 
_user_specified_nameinputs
?
F
*__inference_flatten_7_layer_call_fn_194934

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
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_1913972
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????G>:W S
/
_output_shapes
:?????????G>
 
_user_specified_nameinputs
?
F
*__inference_flatten_3_layer_call_fn_194890

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
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1914292
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
A__inference_HidD1_layer_call_and_return_conditional_losses_191619

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
?
]
A__inference_PoolC_layer_call_and_return_conditional_losses_194353

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????G>*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????G>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
]
A__inference_PoolE_layer_call_and_return_conditional_losses_194620

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????D; *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????D; 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_194420

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
:?????????G>2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????G>*
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
:?????????G>2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????G>2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????G>2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????G>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????G>:W S
/
_output_shapes
:?????????G>
 
_user_specified_nameinputs
?
^
B__inference_PoolA4_layer_call_and_return_conditional_losses_190929

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
]
A__inference_PoolC_layer_call_and_return_conditional_losses_194348

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
E__inference_flatten_9_layer_call_and_return_conditional_losses_191381

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????? 2
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
:?????????G=:W S
/
_output_shapes
:?????????G=
 
_user_specified_nameinputs
?
?
G__inference_concatenate_layer_call_and_return_conditional_losses_195203
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
	inputs_10
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:R N
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
_user_specified_name	inputs/10
??
?"
A__inference_model_layer_call_and_return_conditional_losses_193899

inputs=
#cona_conv2d_readvariableop_resource:2
$cona_biasadd_readvariableop_resource:=
#conb_conv2d_readvariableop_resource:2
$conb_biasadd_readvariableop_resource:>
$cona2_conv2d_readvariableop_resource:3
%cona2_biasadd_readvariableop_resource:=
#conc_conv2d_readvariableop_resource:	2
$conc_biasadd_readvariableop_resource:>
$conb2_conv2d_readvariableop_resource:3
%conb2_biasadd_readvariableop_resource:>
$cona3_conv2d_readvariableop_resource:(3
%cona3_biasadd_readvariableop_resource:(=
#cone_conv2d_readvariableop_resource: 2
$cone_biasadd_readvariableop_resource: =
#cond_conv2d_readvariableop_resource:	2
$cond_biasadd_readvariableop_resource:>
$conc2_conv2d_readvariableop_resource:3
%conc2_biasadd_readvariableop_resource:>
$conb3_conv2d_readvariableop_resource:(3
%conb3_biasadd_readvariableop_resource:(>
$cona4_conv2d_readvariableop_resource:((3
%cona4_biasadd_readvariableop_resource:(8
#hida_matmul_readvariableop_resource:???3
$hida_biasadd_readvariableop_resource:	?9
$hida2_matmul_readvariableop_resource:???4
%hida2_biasadd_readvariableop_resource:	?8
$hida3_matmul_readvariableop_resource:
?F?4
%hida3_biasadd_readvariableop_resource:	?8
$hida4_matmul_readvariableop_resource:
??4
%hida4_biasadd_readvariableop_resource:	?8
#hidb_matmul_readvariableop_resource:ȓ?3
$hidb_biasadd_readvariableop_resource:	?9
$hidb2_matmul_readvariableop_resource:???4
%hidb2_biasadd_readvariableop_resource:	?8
$hidb3_matmul_readvariableop_resource:
?F?4
%hidb3_biasadd_readvariableop_resource:	?8
#hidc_matmul_readvariableop_resource:???3
$hidc_biasadd_readvariableop_resource:	?9
$hidc2_matmul_readvariableop_resource:???4
%hidc2_biasadd_readvariableop_resource:	?9
$hidd1_matmul_readvariableop_resource:???4
%hidd1_biasadd_readvariableop_resource:	?9
$hide1_matmul_readvariableop_resource:???4
%hide1_biasadd_readvariableop_resource:	?7
#hid1_matmul_readvariableop_resource:
??3
$hid1_biasadd_readvariableop_resource:	?7
#hid2_matmul_readvariableop_resource:
??3
$hid2_biasadd_readvariableop_resource:	?9
%output_matmul_readvariableop_resource:
??5
&output_biasadd_readvariableop_resource:	?
identity??ConA/BiasAdd/ReadVariableOp?ConA/Conv2D/ReadVariableOp?ConA2/BiasAdd/ReadVariableOp?ConA2/Conv2D/ReadVariableOp?ConA3/BiasAdd/ReadVariableOp?ConA3/Conv2D/ReadVariableOp?ConA4/BiasAdd/ReadVariableOp?ConA4/Conv2D/ReadVariableOp?ConB/BiasAdd/ReadVariableOp?ConB/Conv2D/ReadVariableOp?ConB2/BiasAdd/ReadVariableOp?ConB2/Conv2D/ReadVariableOp?ConB3/BiasAdd/ReadVariableOp?ConB3/Conv2D/ReadVariableOp?ConC/BiasAdd/ReadVariableOp?ConC/Conv2D/ReadVariableOp?ConC2/BiasAdd/ReadVariableOp?ConC2/Conv2D/ReadVariableOp?ConD/BiasAdd/ReadVariableOp?ConD/Conv2D/ReadVariableOp?ConE/BiasAdd/ReadVariableOp?ConE/Conv2D/ReadVariableOp?Hid1/BiasAdd/ReadVariableOp?Hid1/MatMul/ReadVariableOp?Hid2/BiasAdd/ReadVariableOp?Hid2/MatMul/ReadVariableOp?HidA/BiasAdd/ReadVariableOp?HidA/MatMul/ReadVariableOp?HidA2/BiasAdd/ReadVariableOp?HidA2/MatMul/ReadVariableOp?HidA3/BiasAdd/ReadVariableOp?HidA3/MatMul/ReadVariableOp?HidA4/BiasAdd/ReadVariableOp?HidA4/MatMul/ReadVariableOp?HidB/BiasAdd/ReadVariableOp?HidB/MatMul/ReadVariableOp?HidB2/BiasAdd/ReadVariableOp?HidB2/MatMul/ReadVariableOp?HidB3/BiasAdd/ReadVariableOp?HidB3/MatMul/ReadVariableOp?HidC/BiasAdd/ReadVariableOp?HidC/MatMul/ReadVariableOp?HidC2/BiasAdd/ReadVariableOp?HidC2/MatMul/ReadVariableOp?HidD1/BiasAdd/ReadVariableOp?HidD1/MatMul/ReadVariableOp?HidE1/BiasAdd/ReadVariableOp?HidE1/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
ConA/Conv2D/ReadVariableOpReadVariableOp#cona_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConA/Conv2D/ReadVariableOp?
ConA/Conv2DConv2Dinputs"ConA/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
ConA/Conv2D?
ConA/BiasAdd/ReadVariableOpReadVariableOp$cona_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConA/BiasAdd/ReadVariableOp?
ConA/BiasAddBiasAddConA/Conv2D:output:0#ConA/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
ConA/BiasAddq
	ConA/ReluReluConA/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
	ConA/Relu?
PoolA/MaxPoolMaxPoolConA/Relu:activations:0*/
_output_shapes
:?????????I?*
ksize
*
paddingVALID*
strides
2
PoolA/MaxPool?
ConB/Conv2D/ReadVariableOpReadVariableOp#conb_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConB/Conv2D/ReadVariableOp?
ConB/Conv2DConv2Dinputs"ConB/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
ConB/Conv2D?
ConB/BiasAdd/ReadVariableOpReadVariableOp$conb_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConB/BiasAdd/ReadVariableOp?
ConB/BiasAddBiasAddConB/Conv2D:output:0#ConB/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
ConB/BiasAddq
	ConB/ReluReluConB/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
	ConB/Relus
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
:?????????I?2
dropout/dropout/Mult
dropout/dropout/ShapeShapePoolA/MaxPool:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????I?*
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
:?????????I?2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????I?2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????I?2
dropout/dropout/Mul_1?
PoolB/MaxPoolMaxPoolConB/Relu:activations:0*/
_output_shapes
:?????????H?*
ksize
*
paddingVALID*
strides
2
PoolB/MaxPool?
ConA2/Conv2D/ReadVariableOpReadVariableOp$cona2_conv2d_readvariableop_resource*&
_output_shapes
:*
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
ConC/Conv2D/ReadVariableOpReadVariableOp#conc_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
ConC/Conv2D/ReadVariableOp?
ConC/Conv2DConv2Dinputs"ConC/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
ConC/Conv2D?
ConC/BiasAdd/ReadVariableOpReadVariableOp$conc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConC/BiasAdd/ReadVariableOp?
ConC/BiasAddBiasAddConC/Conv2D:output:0#ConC/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
ConC/BiasAddq
	ConC/ReluReluConC/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
	ConC/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_1/dropout/Const?
dropout_1/dropout/MulMulPoolB/MaxPool:output:0 dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:?????????H?2
dropout_1/dropout/Mulx
dropout_1/dropout/ShapeShapePoolB/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????H?*
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
:?????????H?2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????H?2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????H?2
dropout_1/dropout/Mul_1?
PoolA2/MaxPoolMaxPoolConA2/Relu:activations:0*/
_output_shapes
:?????????#*
ksize
*
paddingVALID*
strides
2
PoolA2/MaxPool?
PoolC/MaxPoolMaxPoolConC/Relu:activations:0*/
_output_shapes
:?????????G>*
ksize
*
paddingVALID*
strides
2
PoolC/MaxPool?
ConB2/Conv2D/ReadVariableOpReadVariableOp$conb2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConB2/Conv2D/ReadVariableOp?
ConB2/Conv2DConv2Ddropout_1/dropout/Mul_1:z:0#ConB2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????F=*
paddingVALID*
strides
2
ConB2/Conv2D?
ConB2/BiasAdd/ReadVariableOpReadVariableOp%conb2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConB2/BiasAdd/ReadVariableOp?
ConB2/BiasAddBiasAddConB2/Conv2D:output:0$ConB2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????F=2
ConB2/BiasAddr

ConB2/ReluReluConB2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????F=2

ConB2/Relu?
ConA3/Conv2D/ReadVariableOpReadVariableOp$cona3_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
ConA3/Conv2D/ReadVariableOp?
ConA3/Conv2DConv2DPoolA2/MaxPool:output:0#ConA3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!(*
paddingVALID*
strides
2
ConA3/Conv2D?
ConA3/BiasAdd/ReadVariableOpReadVariableOp%cona3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
ConA3/BiasAdd/ReadVariableOp?
ConA3/BiasAddBiasAddConA3/Conv2D:output:0$ConA3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!(2
ConA3/BiasAddr

ConA3/ReluReluConA3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????!(2

ConA3/Reluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_2/dropout/Const?
dropout_2/dropout/MulMulPoolC/MaxPool:output:0 dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:?????????G>2
dropout_2/dropout/Mulx
dropout_2/dropout/ShapeShapePoolC/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????G>*
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
:?????????G>2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????G>2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????G>2
dropout_2/dropout/Mul_1?
PoolB2/MaxPoolMaxPoolConB2/Relu:activations:0*/
_output_shapes
:?????????#*
ksize
*
paddingVALID*
strides
2
PoolB2/MaxPool?
PoolA3/MaxPoolMaxPoolConA3/Relu:activations:0*/
_output_shapes
:?????????(*
ksize
*
paddingVALID*
strides
2
PoolA3/MaxPool?
ConE/Conv2D/ReadVariableOpReadVariableOp#cone_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
ConE/Conv2D/ReadVariableOp?
ConE/Conv2DConv2Dinputs"ConE/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
2
ConE/Conv2D?
ConE/BiasAdd/ReadVariableOpReadVariableOp$cone_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
ConE/BiasAdd/ReadVariableOp?
ConE/BiasAddBiasAddConE/Conv2D:output:0#ConE/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
ConE/BiasAddq
	ConE/ReluReluConE/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
	ConE/Relu?
ConD/Conv2D/ReadVariableOpReadVariableOp#cond_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
ConD/Conv2D/ReadVariableOp?
ConD/Conv2DConv2Dinputs"ConD/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
ConD/Conv2D?
ConD/BiasAdd/ReadVariableOpReadVariableOp$cond_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConD/BiasAdd/ReadVariableOp?
ConD/BiasAddBiasAddConD/Conv2D:output:0#ConD/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
ConD/BiasAddq
	ConD/ReluReluConD/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
	ConD/Relu?
ConC2/Conv2D/ReadVariableOpReadVariableOp$conc2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConC2/Conv2D/ReadVariableOp?
ConC2/Conv2DConv2Ddropout_2/dropout/Mul_1:z:0#ConC2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????E<*
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
:?????????E<2
ConC2/BiasAddr

ConC2/ReluReluConC2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????E<2

ConC2/Relu?
ConB3/Conv2D/ReadVariableOpReadVariableOp$conb3_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
ConB3/Conv2D/ReadVariableOp?
ConB3/Conv2DConv2DPoolB2/MaxPool:output:0#ConB3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!(*
paddingVALID*
strides
2
ConB3/Conv2D?
ConB3/BiasAdd/ReadVariableOpReadVariableOp%conb3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
ConB3/BiasAdd/ReadVariableOp?
ConB3/BiasAddBiasAddConB3/Conv2D:output:0$ConB3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!(2
ConB3/BiasAddr

ConB3/ReluReluConB3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????!(2

ConB3/Relu?
ConA4/Conv2D/ReadVariableOpReadVariableOp$cona4_conv2d_readvariableop_resource*&
_output_shapes
:((*
dtype02
ConA4/Conv2D/ReadVariableOp?
ConA4/Conv2DConv2DPoolA3/MaxPool:output:0#ConA4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingVALID*
strides
2
ConA4/Conv2D?
ConA4/BiasAdd/ReadVariableOpReadVariableOp%cona4_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
ConA4/BiasAdd/ReadVariableOp?
ConA4/BiasAddBiasAddConA4/Conv2D:output:0$ConA4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2
ConA4/BiasAddr

ConA4/ReluReluConA4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(2

ConA4/Relu?
PoolE/MaxPoolMaxPoolConE/Relu:activations:0*/
_output_shapes
:?????????D; *
ksize
*
paddingVALID*
strides
2
PoolE/MaxPool?
PoolD/MaxPoolMaxPoolConD/Relu:activations:0*/
_output_shapes
:?????????G=*
ksize
*
paddingVALID*
strides
2
PoolD/MaxPool?
PoolC2/MaxPoolMaxPoolConC2/Relu:activations:0*/
_output_shapes
:?????????"*
ksize
*
paddingVALID*
strides
2
PoolC2/MaxPool?
PoolB3/MaxPoolMaxPoolConB3/Relu:activations:0*/
_output_shapes
:?????????(*
ksize
*
paddingVALID*
strides
2
PoolB3/MaxPool?
PoolA4/MaxPoolMaxPoolConA4/Relu:activations:0*/
_output_shapes
:?????????(*
ksize
*
paddingVALID*
strides
2
PoolA4/MaxPoolw
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_4/dropout/Const?
dropout_4/dropout/MulMulPoolE/MaxPool:output:0 dropout_4/dropout/Const:output:0*
T0*/
_output_shapes
:?????????D; 2
dropout_4/dropout/Mulx
dropout_4/dropout/ShapeShapePoolE/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_4/dropout/Shape?
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????D; *
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
:?????????D; 2 
dropout_4/dropout/GreaterEqual?
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????D; 2
dropout_4/dropout/Cast?
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????D; 2
dropout_4/dropout/Mul_1w
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_3/dropout/Const?
dropout_3/dropout/MulMulPoolD/MaxPool:output:0 dropout_3/dropout/Const:output:0*
T0*/
_output_shapes
:?????????G=2
dropout_3/dropout/Mulx
dropout_3/dropout/ShapeShapePoolD/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shape?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????G=*
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
:?????????G=2 
dropout_3/dropout/GreaterEqual?
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????G=2
dropout_3/dropout/Cast?
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????G=2
dropout_3/dropout/Mul_1y
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_10/dropout/Const?
dropout_10/dropout/MulMulPoolC2/MaxPool:output:0!dropout_10/dropout/Const:output:0*
T0*/
_output_shapes
:?????????"2
dropout_10/dropout/Mul{
dropout_10/dropout/ShapeShapePoolC2/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_10/dropout/Shape?
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????"*
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
:?????????"2!
dropout_10/dropout/GreaterEqual?
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????"2
dropout_10/dropout/Cast?
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????"2
dropout_10/dropout/Mul_1w
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_9/dropout/Const?
dropout_9/dropout/MulMulPoolB3/MaxPool:output:0 dropout_9/dropout/Const:output:0*
T0*/
_output_shapes
:?????????(2
dropout_9/dropout/Muly
dropout_9/dropout/ShapeShapePoolB3/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shape?
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2 
dropout_9/dropout/GreaterEqual?
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????(2
dropout_9/dropout/Cast?
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????(2
dropout_9/dropout/Mul_1w
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_8/dropout/Const?
dropout_8/dropout/MulMulPoolB2/MaxPool:output:0 dropout_8/dropout/Const:output:0*
T0*/
_output_shapes
:?????????#2
dropout_8/dropout/Muly
dropout_8/dropout/ShapeShapePoolB2/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_8/dropout/Shape?
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????#*
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
:?????????#2 
dropout_8/dropout/GreaterEqual?
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????#2
dropout_8/dropout/Cast?
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????#2
dropout_8/dropout/Mul_1w
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_7/dropout/Const?
dropout_7/dropout/MulMulPoolA4/MaxPool:output:0 dropout_7/dropout/Const:output:0*
T0*/
_output_shapes
:?????????(2
dropout_7/dropout/Muly
dropout_7/dropout/ShapeShapePoolA4/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_7/dropout/Shape?
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2 
dropout_7/dropout/GreaterEqual?
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????(2
dropout_7/dropout/Cast?
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????(2
dropout_7/dropout/Mul_1w
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_6/dropout/Const?
dropout_6/dropout/MulMulPoolA3/MaxPool:output:0 dropout_6/dropout/Const:output:0*
T0*/
_output_shapes
:?????????(2
dropout_6/dropout/Muly
dropout_6/dropout/ShapeShapePoolA3/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_6/dropout/Shape?
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2 
dropout_6/dropout/GreaterEqual?
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????(2
dropout_6/dropout/Cast?
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????(2
dropout_6/dropout/Mul_1w
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_5/dropout/Const?
dropout_5/dropout/MulMulPoolA2/MaxPool:output:0 dropout_5/dropout/Const:output:0*
T0*/
_output_shapes
:?????????#2
dropout_5/dropout/Muly
dropout_5/dropout/ShapeShapePoolA2/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_5/dropout/Shape?
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????#*
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
:?????????#2 
dropout_5/dropout/GreaterEqual?
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????#2
dropout_5/dropout/Cast?
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????#2
dropout_5/dropout/Mul_1u
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????? 2
flatten_10/Const?
flatten_10/ReshapeReshapedropout_4/dropout/Mul_1:z:0flatten_10/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_10/Reshapes
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????? 2
flatten_9/Const?
flatten_9/ReshapeReshapedropout_3/dropout/Mul_1:z:0flatten_9/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_9/Reshapes
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????O  2
flatten_8/Const?
flatten_8/ReshapeReshapedropout_10/dropout/Mul_1:z:0flatten_8/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_8/Reshapes
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????W 2
flatten_7/Const?
flatten_7/ReshapeReshapedropout_2/dropout/Mul_1:z:0flatten_7/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_7/Reshapes
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? #  2
flatten_6/Const?
flatten_6/ReshapeReshapedropout_9/dropout/Mul_1:z:0flatten_6/Const:output:0*
T0*(
_output_shapes
:??????????F2
flatten_6/Reshapes
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????R  2
flatten_5/Const?
flatten_5/ReshapeReshapedropout_8/dropout/Mul_1:z:0flatten_5/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_5/Reshapes
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????	 2
flatten_4/Const?
flatten_4/ReshapeReshapePoolB/MaxPool:output:0flatten_4/Const:output:0*
T0*)
_output_shapes
:?????????ȓ2
flatten_4/Reshapes
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_3/Const?
flatten_3/ReshapeReshapedropout_7/dropout/Mul_1:z:0flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_3/Reshapes
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? #  2
flatten_2/Const?
flatten_2/ReshapeReshapedropout_6/dropout/Mul_1:z:0flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????F2
flatten_2/Reshapes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????R  2
flatten_1/Const?
flatten_1/ReshapeReshapedropout_5/dropout/Mul_1:z:0flatten_1/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_1/Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"??????  2
flatten/Const?
flatten/ReshapeReshapePoolA/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten/Reshape?
HidA/MatMul/ReadVariableOpReadVariableOp#hida_matmul_readvariableop_resource*!
_output_shapes
:???*
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
?F?*
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
??*
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
HidB/MatMul/ReadVariableOpReadVariableOp#hidb_matmul_readvariableop_resource*!
_output_shapes
:ȓ?*
dtype02
HidB/MatMul/ReadVariableOp?
HidB/MatMulMatMulflatten_4/Reshape:output:0"HidB/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidB/MatMul?
HidB/BiasAdd/ReadVariableOpReadVariableOp$hidb_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidB/BiasAdd/ReadVariableOp?
HidB/BiasAddBiasAddHidB/MatMul:product:0#HidB/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidB/BiasAddh
	HidB/ReluReluHidB/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
	HidB/Relu?
HidB2/MatMul/ReadVariableOpReadVariableOp$hidb2_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
HidB2/MatMul/ReadVariableOp?
HidB2/MatMulMatMulflatten_5/Reshape:output:0#HidB2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidB2/MatMul?
HidB2/BiasAdd/ReadVariableOpReadVariableOp%hidb2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidB2/BiasAdd/ReadVariableOp?
HidB2/BiasAddBiasAddHidB2/MatMul:product:0$HidB2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidB2/BiasAddk

HidB2/ReluReluHidB2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidB2/Relu?
HidB3/MatMul/ReadVariableOpReadVariableOp$hidb3_matmul_readvariableop_resource* 
_output_shapes
:
?F?*
dtype02
HidB3/MatMul/ReadVariableOp?
HidB3/MatMulMatMulflatten_6/Reshape:output:0#HidB3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidB3/MatMul?
HidB3/BiasAdd/ReadVariableOpReadVariableOp%hidb3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidB3/BiasAdd/ReadVariableOp?
HidB3/BiasAddBiasAddHidB3/MatMul:product:0$HidB3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidB3/BiasAddk

HidB3/ReluReluHidB3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidB3/Relu?
HidC/MatMul/ReadVariableOpReadVariableOp#hidc_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
HidC/MatMul/ReadVariableOp?
HidC/MatMulMatMulflatten_7/Reshape:output:0"HidC/MatMul/ReadVariableOp:value:0*
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
HidC2/MatMulMatMulflatten_8/Reshape:output:0#HidC2/MatMul/ReadVariableOp:value:0*
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
HidD1/MatMul/ReadVariableOpReadVariableOp$hidd1_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
HidD1/MatMul/ReadVariableOp?
HidD1/MatMulMatMulflatten_9/Reshape:output:0#HidD1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidD1/MatMul?
HidD1/BiasAdd/ReadVariableOpReadVariableOp%hidd1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidD1/BiasAdd/ReadVariableOp?
HidD1/BiasAddBiasAddHidD1/MatMul:product:0$HidD1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidD1/BiasAddk

HidD1/ReluReluHidD1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidD1/Relu?
HidE1/MatMul/ReadVariableOpReadVariableOp$hide1_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
HidE1/MatMul/ReadVariableOp?
HidE1/MatMulMatMulflatten_10/Reshape:output:0#HidE1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidE1/MatMul?
HidE1/BiasAdd/ReadVariableOpReadVariableOp%hide1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidE1/BiasAdd/ReadVariableOp?
HidE1/BiasAddBiasAddHidE1/MatMul:product:0$HidE1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidE1/BiasAddk

HidE1/ReluReluHidE1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidE1/Relut
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2HidA/Relu:activations:0HidA2/Relu:activations:0HidA3/Relu:activations:0HidA4/Relu:activations:0HidB/Relu:activations:0HidB2/Relu:activations:0HidB3/Relu:activations:0HidC/Relu:activations:0HidC2/Relu:activations:0HidD1/Relu:activations:0HidE1/Relu:activations:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate/concat?
Hid1/MatMul/ReadVariableOpReadVariableOp#hid1_matmul_readvariableop_resource* 
_output_shapes
:
??*
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
_output_shapes
:
??*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMulHid2/Tanh:y:0$output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
output/BiasAddw
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
output/Softmaxt
IdentityIdentityoutput/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^ConA/BiasAdd/ReadVariableOp^ConA/Conv2D/ReadVariableOp^ConA2/BiasAdd/ReadVariableOp^ConA2/Conv2D/ReadVariableOp^ConA3/BiasAdd/ReadVariableOp^ConA3/Conv2D/ReadVariableOp^ConA4/BiasAdd/ReadVariableOp^ConA4/Conv2D/ReadVariableOp^ConB/BiasAdd/ReadVariableOp^ConB/Conv2D/ReadVariableOp^ConB2/BiasAdd/ReadVariableOp^ConB2/Conv2D/ReadVariableOp^ConB3/BiasAdd/ReadVariableOp^ConB3/Conv2D/ReadVariableOp^ConC/BiasAdd/ReadVariableOp^ConC/Conv2D/ReadVariableOp^ConC2/BiasAdd/ReadVariableOp^ConC2/Conv2D/ReadVariableOp^ConD/BiasAdd/ReadVariableOp^ConD/Conv2D/ReadVariableOp^ConE/BiasAdd/ReadVariableOp^ConE/Conv2D/ReadVariableOp^Hid1/BiasAdd/ReadVariableOp^Hid1/MatMul/ReadVariableOp^Hid2/BiasAdd/ReadVariableOp^Hid2/MatMul/ReadVariableOp^HidA/BiasAdd/ReadVariableOp^HidA/MatMul/ReadVariableOp^HidA2/BiasAdd/ReadVariableOp^HidA2/MatMul/ReadVariableOp^HidA3/BiasAdd/ReadVariableOp^HidA3/MatMul/ReadVariableOp^HidA4/BiasAdd/ReadVariableOp^HidA4/MatMul/ReadVariableOp^HidB/BiasAdd/ReadVariableOp^HidB/MatMul/ReadVariableOp^HidB2/BiasAdd/ReadVariableOp^HidB2/MatMul/ReadVariableOp^HidB3/BiasAdd/ReadVariableOp^HidB3/MatMul/ReadVariableOp^HidC/BiasAdd/ReadVariableOp^HidC/MatMul/ReadVariableOp^HidC2/BiasAdd/ReadVariableOp^HidC2/MatMul/ReadVariableOp^HidD1/BiasAdd/ReadVariableOp^HidD1/MatMul/ReadVariableOp^HidE1/BiasAdd/ReadVariableOp^HidE1/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
ConA/BiasAdd/ReadVariableOpConA/BiasAdd/ReadVariableOp28
ConA/Conv2D/ReadVariableOpConA/Conv2D/ReadVariableOp2<
ConA2/BiasAdd/ReadVariableOpConA2/BiasAdd/ReadVariableOp2:
ConA2/Conv2D/ReadVariableOpConA2/Conv2D/ReadVariableOp2<
ConA3/BiasAdd/ReadVariableOpConA3/BiasAdd/ReadVariableOp2:
ConA3/Conv2D/ReadVariableOpConA3/Conv2D/ReadVariableOp2<
ConA4/BiasAdd/ReadVariableOpConA4/BiasAdd/ReadVariableOp2:
ConA4/Conv2D/ReadVariableOpConA4/Conv2D/ReadVariableOp2:
ConB/BiasAdd/ReadVariableOpConB/BiasAdd/ReadVariableOp28
ConB/Conv2D/ReadVariableOpConB/Conv2D/ReadVariableOp2<
ConB2/BiasAdd/ReadVariableOpConB2/BiasAdd/ReadVariableOp2:
ConB2/Conv2D/ReadVariableOpConB2/Conv2D/ReadVariableOp2<
ConB3/BiasAdd/ReadVariableOpConB3/BiasAdd/ReadVariableOp2:
ConB3/Conv2D/ReadVariableOpConB3/Conv2D/ReadVariableOp2:
ConC/BiasAdd/ReadVariableOpConC/BiasAdd/ReadVariableOp28
ConC/Conv2D/ReadVariableOpConC/Conv2D/ReadVariableOp2<
ConC2/BiasAdd/ReadVariableOpConC2/BiasAdd/ReadVariableOp2:
ConC2/Conv2D/ReadVariableOpConC2/Conv2D/ReadVariableOp2:
ConD/BiasAdd/ReadVariableOpConD/BiasAdd/ReadVariableOp28
ConD/Conv2D/ReadVariableOpConD/Conv2D/ReadVariableOp2:
ConE/BiasAdd/ReadVariableOpConE/BiasAdd/ReadVariableOp28
ConE/Conv2D/ReadVariableOpConE/Conv2D/ReadVariableOp2:
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
HidB/BiasAdd/ReadVariableOpHidB/BiasAdd/ReadVariableOp28
HidB/MatMul/ReadVariableOpHidB/MatMul/ReadVariableOp2<
HidB2/BiasAdd/ReadVariableOpHidB2/BiasAdd/ReadVariableOp2:
HidB2/MatMul/ReadVariableOpHidB2/MatMul/ReadVariableOp2<
HidB3/BiasAdd/ReadVariableOpHidB3/BiasAdd/ReadVariableOp2:
HidB3/MatMul/ReadVariableOpHidB3/MatMul/ReadVariableOp2:
HidC/BiasAdd/ReadVariableOpHidC/BiasAdd/ReadVariableOp28
HidC/MatMul/ReadVariableOpHidC/MatMul/ReadVariableOp2<
HidC2/BiasAdd/ReadVariableOpHidC2/BiasAdd/ReadVariableOp2:
HidC2/MatMul/ReadVariableOpHidC2/MatMul/ReadVariableOp2<
HidD1/BiasAdd/ReadVariableOpHidD1/BiasAdd/ReadVariableOp2:
HidD1/MatMul/ReadVariableOpHidD1/MatMul/ReadVariableOp2<
HidE1/BiasAdd/ReadVariableOpHidE1/BiasAdd/ReadVariableOp2:
HidE1/MatMul/ReadVariableOpHidE1/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_8_layer_call_and_return_conditional_losses_192126

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
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_194154

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????I?2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????I?2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????I?:W S
/
_output_shapes
:?????????I?
 
_user_specified_nameinputs
?
?
&__inference_ConA2_layer_call_fn_194216

inputs!
unknown:
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
A__inference_ConA2_layer_call_and_return_conditional_losses_1911012
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
:?????????I?: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????I?
 
_user_specified_nameinputs
?
?
A__inference_ConC2_layer_call_and_return_conditional_losses_191241

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
:?????????E<*
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
:?????????E<2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????E<2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????E<2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????G>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????G>
 
_user_specified_nameinputs
?
^
B__inference_PoolA2_layer_call_and_return_conditional_losses_194246

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
E__inference_dropout_2_layer_call_and_return_conditional_losses_192326

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
:?????????G>2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????G>*
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
:?????????G>2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????G>2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????G>2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????G>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????G>:W S
/
_output_shapes
:?????????G>
 
_user_specified_nameinputs
?
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_194797

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????G=2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????G=2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????G=:W S
/
_output_shapes
:?????????G=
 
_user_specified_nameinputs
?
F
*__inference_flatten_4_layer_call_fn_194901

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
:?????????ȓ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_1914212
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:?????????ȓ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????H?:W S
/
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?
?
A__inference_HidB2_layer_call_and_return_conditional_losses_191551

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
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_194885

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_191437

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? #  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????F2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????F2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
d
E__inference_dropout_9_layer_call_and_return_conditional_losses_192149

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
:?????????(2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????(2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????(2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
C
'__inference_PoolA3_layer_call_fn_194383

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
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA3_layer_call_and_return_conditional_losses_1911942
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????!(:W S
/
_output_shapes
:?????????!(
 
_user_specified_nameinputs
?
^
B__inference_PoolB3_layer_call_and_return_conditional_losses_190951

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
D
(__inference_flatten_layer_call_fn_194857

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
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1914532
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????I?:W S
/
_output_shapes
:?????????I?
 
_user_specified_nameinputs
?
]
A__inference_PoolB_layer_call_and_return_conditional_losses_190819

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
^
B__inference_PoolB2_layer_call_and_return_conditional_losses_194393

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
:?????????F=:W S
/
_output_shapes
:?????????F=
 
_user_specified_nameinputs
?
?
%__inference_ConA_layer_call_fn_194129

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConA_layer_call_and_return_conditional_losses_1910482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

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
@__inference_ConA_layer_call_and_return_conditional_losses_194120

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????2

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
?
A__inference_HidE1_layer_call_and_return_conditional_losses_191636

inputs3
matmul_readvariableop_resource:???.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
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
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_6_layer_call_and_return_conditional_losses_194674

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
:?????????(2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????(2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????(2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_194716

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
?
c
*__inference_dropout_9_layer_call_fn_194765

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
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_1921492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
??
?
A__inference_model_layer_call_and_return_conditional_losses_192725

inputs%
cona_192565:
cona_192567:%
conb_192571:
conb_192573:&
cona2_192578:
cona2_192580:%
conc_192583:	
conc_192585:&
conb2_192591:
conb2_192593:&
cona3_192596:(
cona3_192598:(%
cone_192604: 
cone_192606: %
cond_192609:	
cond_192611:&
conc2_192614:
conc2_192616:&
conb3_192619:(
conb3_192621:(&
cona4_192624:((
cona4_192626:( 
hida_192653:???
hida_192655:	?!
hida2_192658:???
hida2_192660:	? 
hida3_192663:
?F?
hida3_192665:	? 
hida4_192668:
??
hida4_192670:	? 
hidb_192673:ȓ?
hidb_192675:	?!
hidb2_192678:???
hidb2_192680:	? 
hidb3_192683:
?F?
hidb3_192685:	? 
hidc_192688:???
hidc_192690:	?!
hidc2_192693:???
hidc2_192695:	?!
hidd1_192698:???
hidd1_192700:	?!
hide1_192703:???
hide1_192705:	?
hid1_192709:
??
hid1_192711:	?
hid2_192714:
??
hid2_192716:	?!
output_192719:
??
output_192721:	?
identity??ConA/StatefulPartitionedCall?ConA2/StatefulPartitionedCall?ConA3/StatefulPartitionedCall?ConA4/StatefulPartitionedCall?ConB/StatefulPartitionedCall?ConB2/StatefulPartitionedCall?ConB3/StatefulPartitionedCall?ConC/StatefulPartitionedCall?ConC2/StatefulPartitionedCall?ConD/StatefulPartitionedCall?ConE/StatefulPartitionedCall?Hid1/StatefulPartitionedCall?Hid2/StatefulPartitionedCall?HidA/StatefulPartitionedCall?HidA2/StatefulPartitionedCall?HidA3/StatefulPartitionedCall?HidA4/StatefulPartitionedCall?HidB/StatefulPartitionedCall?HidB2/StatefulPartitionedCall?HidB3/StatefulPartitionedCall?HidC/StatefulPartitionedCall?HidC2/StatefulPartitionedCall?HidD1/StatefulPartitionedCall?HidE1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?"dropout_10/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?!dropout_5/StatefulPartitionedCall?!dropout_6/StatefulPartitionedCall?!dropout_7/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?!dropout_9/StatefulPartitionedCall?output/StatefulPartitionedCall?
ConA/StatefulPartitionedCallStatefulPartitionedCallinputscona_192565cona_192567*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConA_layer_call_and_return_conditional_losses_1910482
ConA/StatefulPartitionedCall?
PoolA/PartitionedCallPartitionedCall%ConA/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolA_layer_call_and_return_conditional_losses_1910582
PoolA/PartitionedCall?
ConB/StatefulPartitionedCallStatefulPartitionedCallinputsconb_192571conb_192573*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConB_layer_call_and_return_conditional_losses_1910712
ConB/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCallPoolA/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1924272!
dropout/StatefulPartitionedCall?
PoolB/PartitionedCallPartitionedCall%ConB/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????H?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolB_layer_call_and_return_conditional_losses_1910882
PoolB/PartitionedCall?
ConA2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0cona2_192578cona2_192580*
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
A__inference_ConA2_layer_call_and_return_conditional_losses_1911012
ConA2/StatefulPartitionedCall?
ConC/StatefulPartitionedCallStatefulPartitionedCallinputsconc_192583conc_192585*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConC_layer_call_and_return_conditional_losses_1911182
ConC/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCallPoolB/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????H?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1923792#
!dropout_1/StatefulPartitionedCall?
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
B__inference_PoolA2_layer_call_and_return_conditional_losses_1911352
PoolA2/PartitionedCall?
PoolC/PartitionedCallPartitionedCall%ConC/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????G>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolC_layer_call_and_return_conditional_losses_1911412
PoolC/PartitionedCall?
ConB2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conb2_192591conb2_192593*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConB2_layer_call_and_return_conditional_losses_1911542
ConB2/StatefulPartitionedCall?
ConA3/StatefulPartitionedCallStatefulPartitionedCallPoolA2/PartitionedCall:output:0cona3_192596cona3_192598*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConA3_layer_call_and_return_conditional_losses_1911712
ConA3/StatefulPartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCallPoolC/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????G>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_1923262#
!dropout_2/StatefulPartitionedCall?
PoolB2/PartitionedCallPartitionedCall&ConB2/StatefulPartitionedCall:output:0*
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
B__inference_PoolB2_layer_call_and_return_conditional_losses_1911882
PoolB2/PartitionedCall?
PoolA3/PartitionedCallPartitionedCall&ConA3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA3_layer_call_and_return_conditional_losses_1911942
PoolA3/PartitionedCall?
ConE/StatefulPartitionedCallStatefulPartitionedCallinputscone_192604cone_192606*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConE_layer_call_and_return_conditional_losses_1912072
ConE/StatefulPartitionedCall?
ConD/StatefulPartitionedCallStatefulPartitionedCallinputscond_192609cond_192611*
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
@__inference_ConD_layer_call_and_return_conditional_losses_1912242
ConD/StatefulPartitionedCall?
ConC2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0conc2_192614conc2_192616*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????E<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConC2_layer_call_and_return_conditional_losses_1912412
ConC2/StatefulPartitionedCall?
ConB3/StatefulPartitionedCallStatefulPartitionedCallPoolB2/PartitionedCall:output:0conb3_192619conb3_192621*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConB3_layer_call_and_return_conditional_losses_1912582
ConB3/StatefulPartitionedCall?
ConA4/StatefulPartitionedCallStatefulPartitionedCallPoolA3/PartitionedCall:output:0cona4_192624cona4_192626*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_ConA4_layer_call_and_return_conditional_losses_1912752
ConA4/StatefulPartitionedCall?
PoolE/PartitionedCallPartitionedCall%ConE/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????D; * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolE_layer_call_and_return_conditional_losses_1912852
PoolE/PartitionedCall?
PoolD/PartitionedCallPartitionedCall%ConD/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????G=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolD_layer_call_and_return_conditional_losses_1912912
PoolD/PartitionedCall?
PoolC2/PartitionedCallPartitionedCall&ConC2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolC2_layer_call_and_return_conditional_losses_1912972
PoolC2/PartitionedCall?
PoolB3/PartitionedCallPartitionedCall&ConB3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolB3_layer_call_and_return_conditional_losses_1913032
PoolB3/PartitionedCall?
PoolA4/PartitionedCallPartitionedCall&ConA4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_PoolA4_layer_call_and_return_conditional_losses_1913092
PoolA4/PartitionedCall?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCallPoolE/PartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????D; * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1922182#
!dropout_4/StatefulPartitionedCall?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCallPoolD/PartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????G=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_1921952#
!dropout_3/StatefulPartitionedCall?
"dropout_10/StatefulPartitionedCallStatefulPartitionedCallPoolC2/PartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_1921722$
"dropout_10/StatefulPartitionedCall?
!dropout_9/StatefulPartitionedCallStatefulPartitionedCallPoolB3/PartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_1921492#
!dropout_9/StatefulPartitionedCall?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCallPoolB2/PartitionedCall:output:0"^dropout_9/StatefulPartitionedCall*
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
E__inference_dropout_8_layer_call_and_return_conditional_losses_1921262#
!dropout_8/StatefulPartitionedCall?
!dropout_7/StatefulPartitionedCallStatefulPartitionedCallPoolA4/PartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_1921032#
!dropout_7/StatefulPartitionedCall?
!dropout_6/StatefulPartitionedCallStatefulPartitionedCallPoolA3/PartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_1920802#
!dropout_6/StatefulPartitionedCall?
!dropout_5/StatefulPartitionedCallStatefulPartitionedCallPoolA2/PartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
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
E__inference_dropout_5_layer_call_and_return_conditional_losses_1920572#
!dropout_5/StatefulPartitionedCall?
flatten_10/PartitionedCallPartitionedCall*dropout_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_10_layer_call_and_return_conditional_losses_1913732
flatten_10/PartitionedCall?
flatten_9/PartitionedCallPartitionedCall*dropout_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *N
fIRG
E__inference_flatten_9_layer_call_and_return_conditional_losses_1913812
flatten_9/PartitionedCall?
flatten_8/PartitionedCallPartitionedCall+dropout_10/StatefulPartitionedCall:output:0*
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
E__inference_flatten_8_layer_call_and_return_conditional_losses_1913892
flatten_8/PartitionedCall?
flatten_7/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_1913972
flatten_7/PartitionedCall?
flatten_6/PartitionedCallPartitionedCall*dropout_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????F* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_1914052
flatten_6/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall*dropout_8/StatefulPartitionedCall:output:0*
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
E__inference_flatten_5_layer_call_and_return_conditional_losses_1914132
flatten_5/PartitionedCall?
flatten_4/PartitionedCallPartitionedCallPoolB/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????ȓ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_1914212
flatten_4/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall*dropout_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1914292
flatten_3/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall*dropout_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????F* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_1914372
flatten_2/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall*dropout_5/StatefulPartitionedCall:output:0*
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
E__inference_flatten_1_layer_call_and_return_conditional_losses_1914452
flatten_1/PartitionedCall?
flatten/PartitionedCallPartitionedCallPoolA/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1914532
flatten/PartitionedCall?
HidA/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0hida_192653hida_192655*
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
@__inference_HidA_layer_call_and_return_conditional_losses_1914662
HidA/StatefulPartitionedCall?
HidA2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0hida2_192658hida2_192660*
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
A__inference_HidA2_layer_call_and_return_conditional_losses_1914832
HidA2/StatefulPartitionedCall?
HidA3/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0hida3_192663hida3_192665*
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
A__inference_HidA3_layer_call_and_return_conditional_losses_1915002
HidA3/StatefulPartitionedCall?
HidA4/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0hida4_192668hida4_192670*
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
A__inference_HidA4_layer_call_and_return_conditional_losses_1915172
HidA4/StatefulPartitionedCall?
HidB/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0hidb_192673hidb_192675*
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
@__inference_HidB_layer_call_and_return_conditional_losses_1915342
HidB/StatefulPartitionedCall?
HidB2/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0hidb2_192678hidb2_192680*
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
A__inference_HidB2_layer_call_and_return_conditional_losses_1915512
HidB2/StatefulPartitionedCall?
HidB3/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0hidb3_192683hidb3_192685*
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
A__inference_HidB3_layer_call_and_return_conditional_losses_1915682
HidB3/StatefulPartitionedCall?
HidC/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0hidc_192688hidc_192690*
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
@__inference_HidC_layer_call_and_return_conditional_losses_1915852
HidC/StatefulPartitionedCall?
HidC2/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0hidc2_192693hidc2_192695*
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
A__inference_HidC2_layer_call_and_return_conditional_losses_1916022
HidC2/StatefulPartitionedCall?
HidD1/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0hidd1_192698hidd1_192700*
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
A__inference_HidD1_layer_call_and_return_conditional_losses_1916192
HidD1/StatefulPartitionedCall?
HidE1/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0hide1_192703hide1_192705*
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
A__inference_HidE1_layer_call_and_return_conditional_losses_1916362
HidE1/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall%HidA/StatefulPartitionedCall:output:0&HidA2/StatefulPartitionedCall:output:0&HidA3/StatefulPartitionedCall:output:0&HidA4/StatefulPartitionedCall:output:0%HidB/StatefulPartitionedCall:output:0&HidB2/StatefulPartitionedCall:output:0&HidB3/StatefulPartitionedCall:output:0%HidC/StatefulPartitionedCall:output:0&HidC2/StatefulPartitionedCall:output:0&HidD1/StatefulPartitionedCall:output:0&HidE1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_1916582
concatenate/PartitionedCall?
Hid1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0hid1_192709hid1_192711*
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
@__inference_Hid1_layer_call_and_return_conditional_losses_1916712
Hid1/StatefulPartitionedCall?
Hid2/StatefulPartitionedCallStatefulPartitionedCall%Hid1/StatefulPartitionedCall:output:0hid2_192714hid2_192716*
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
@__inference_Hid2_layer_call_and_return_conditional_losses_1916882
Hid2/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall%Hid2/StatefulPartitionedCall:output:0output_192719output_192721*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1917052 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?	
NoOpNoOp^ConA/StatefulPartitionedCall^ConA2/StatefulPartitionedCall^ConA3/StatefulPartitionedCall^ConA4/StatefulPartitionedCall^ConB/StatefulPartitionedCall^ConB2/StatefulPartitionedCall^ConB3/StatefulPartitionedCall^ConC/StatefulPartitionedCall^ConC2/StatefulPartitionedCall^ConD/StatefulPartitionedCall^ConE/StatefulPartitionedCall^Hid1/StatefulPartitionedCall^Hid2/StatefulPartitionedCall^HidA/StatefulPartitionedCall^HidA2/StatefulPartitionedCall^HidA3/StatefulPartitionedCall^HidA4/StatefulPartitionedCall^HidB/StatefulPartitionedCall^HidB2/StatefulPartitionedCall^HidB3/StatefulPartitionedCall^HidC/StatefulPartitionedCall^HidC2/StatefulPartitionedCall^HidD1/StatefulPartitionedCall^HidE1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
ConA/StatefulPartitionedCallConA/StatefulPartitionedCall2>
ConA2/StatefulPartitionedCallConA2/StatefulPartitionedCall2>
ConA3/StatefulPartitionedCallConA3/StatefulPartitionedCall2>
ConA4/StatefulPartitionedCallConA4/StatefulPartitionedCall2<
ConB/StatefulPartitionedCallConB/StatefulPartitionedCall2>
ConB2/StatefulPartitionedCallConB2/StatefulPartitionedCall2>
ConB3/StatefulPartitionedCallConB3/StatefulPartitionedCall2<
ConC/StatefulPartitionedCallConC/StatefulPartitionedCall2>
ConC2/StatefulPartitionedCallConC2/StatefulPartitionedCall2<
ConD/StatefulPartitionedCallConD/StatefulPartitionedCall2<
ConE/StatefulPartitionedCallConE/StatefulPartitionedCall2<
Hid1/StatefulPartitionedCallHid1/StatefulPartitionedCall2<
Hid2/StatefulPartitionedCallHid2/StatefulPartitionedCall2<
HidA/StatefulPartitionedCallHidA/StatefulPartitionedCall2>
HidA2/StatefulPartitionedCallHidA2/StatefulPartitionedCall2>
HidA3/StatefulPartitionedCallHidA3/StatefulPartitionedCall2>
HidA4/StatefulPartitionedCallHidA4/StatefulPartitionedCall2<
HidB/StatefulPartitionedCallHidB/StatefulPartitionedCall2>
HidB2/StatefulPartitionedCallHidB2/StatefulPartitionedCall2>
HidB3/StatefulPartitionedCallHidB3/StatefulPartitionedCall2<
HidC/StatefulPartitionedCallHidC/StatefulPartitionedCall2>
HidC2/StatefulPartitionedCallHidC2/StatefulPartitionedCall2>
HidD1/StatefulPartitionedCallHidD1/StatefulPartitionedCall2>
HidE1/StatefulPartitionedCallHidE1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2F
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
?
?
&__inference_HidE1_layer_call_fn_195187

inputs
unknown:???
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
A__inference_HidE1_layer_call_and_return_conditional_losses_1916362
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
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
@__inference_ConE_layer_call_and_return_conditional_losses_194521

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:??????????? 2

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
a
E__inference_flatten_7_layer_call_and_return_conditional_losses_191397

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????W 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????G>:W S
/
_output_shapes
:?????????G>
 
_user_specified_nameinputs
?
^
B__inference_PoolA4_layer_call_and_return_conditional_losses_194540

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????(*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_194273

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
:?????????H?2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????H?*
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
:?????????H?2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????H?2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????H?2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????H?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????H?:W S
/
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?
d
E__inference_dropout_4_layer_call_and_return_conditional_losses_194836

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
:?????????D; 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????D; *
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
:?????????D; 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????D; 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????D; 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????D; 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????D; :W S
/
_output_shapes
:?????????D; 
 
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
serving_default_input:0???????????;
output1
StatefulPartitionedCall:0??????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
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
layer_with_weights-9
layer-19
layer_with_weights-10
layer-20
layer-21
layer-22
layer-23
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
.layer_with_weights-11
.layer-45
/layer_with_weights-12
/layer-46
0layer_with_weights-13
0layer-47
1layer_with_weights-14
1layer-48
2layer_with_weights-15
2layer-49
3layer_with_weights-16
3layer-50
4layer_with_weights-17
4layer-51
5layer_with_weights-18
5layer-52
6layer_with_weights-19
6layer-53
7layer_with_weights-20
7layer-54
8layer_with_weights-21
8layer-55
9layer-56
:layer_with_weights-22
:layer-57
;layer_with_weights-23
;layer-58
<layer_with_weights-24
<layer-59
=	optimizer
>regularization_losses
?trainable_variables
@	variables
A	keras_api
B
signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"
_tf_keras_network
"
_tf_keras_input_layer
?

Ckernel
Dbias
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Qkernel
Rbias
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Wkernel
Xbias
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
]regularization_losses
^trainable_variables
_	variables
`	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
aregularization_losses
btrainable_variables
c	variables
d	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
eregularization_losses
ftrainable_variables
g	variables
h	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

ikernel
jbias
kregularization_losses
ltrainable_variables
m	variables
n	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

okernel
pbias
qregularization_losses
rtrainable_variables
s	variables
t	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

ukernel
vbias
wregularization_losses
xtrainable_variables
y	variables
z	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
{regularization_losses
|trainable_variables
}	variables
~	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?	
	?iter
?beta_1
?beta_2

?decay
?learning_rateCm?Dm?Qm?Rm?Wm?Xm?im?jm?om?pm?um?vm?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?Cv?Dv?Qv?Rv?Wv?Xv?iv?jv?ov?pv?uv?vv?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
 "
trackable_list_wrapper
?
C0
D1
Q2
R3
W4
X5
i6
j7
o8
p9
u10
v11
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
?22
?23
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
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49"
trackable_list_wrapper
?
C0
D1
Q2
R3
W4
X5
i6
j7
o8
p9
u10
v11
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
?22
?23
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
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
>regularization_losses
?trainable_variables
@	variables
?metrics
?layers
?non_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
%:#2ConA/kernel
:2	ConA/bias
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
Eregularization_losses
Ftrainable_variables
G	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
Iregularization_losses
Jtrainable_variables
K	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
Mregularization_losses
Ntrainable_variables
O	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#2ConB/kernel
:2	ConB/bias
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
Sregularization_losses
Ttrainable_variables
U	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$2ConA2/kernel
:2
ConA2/bias
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
Yregularization_losses
Ztrainable_variables
[	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
]regularization_losses
^trainable_variables
_	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
aregularization_losses
btrainable_variables
c	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
eregularization_losses
ftrainable_variables
g	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#	2ConC/kernel
:2	ConC/bias
 "
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
kregularization_losses
ltrainable_variables
m	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$(2ConA3/kernel
:(2
ConA3/bias
 "
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
qregularization_losses
rtrainable_variables
s	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$2ConB2/kernel
:2
ConB2/bias
 "
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
wregularization_losses
xtrainable_variables
y	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
{regularization_losses
|trainable_variables
}	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$((2ConA4/kernel
:(2
ConA4/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$(2ConB3/kernel
:(2
ConB3/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$2ConC2/kernel
:2
ConC2/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#	2ConD/kernel
:2	ConD/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:# 2ConE/kernel
: 2	ConE/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :???2HidA/kernel
:?2	HidA/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:???2HidA2/kernel
:?2
HidA2/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :
?F?2HidA3/kernel
:?2
HidA3/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :
??2HidA4/kernel
:?2
HidA4/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :ȓ?2HidB/kernel
:?2	HidB/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:???2HidB2/kernel
:?2
HidB2/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :
?F?2HidB3/kernel
:?2
HidB3/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :???2HidC/kernel
:?2	HidC/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:???2HidC2/kernel
:?2
HidC2/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:???2HidD1/kernel
:?2
HidD1/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:???2HidE1/kernel
:?2
HidE1/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
??2Hid1/kernel
:?2	Hid1/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
??2Hid2/kernel
:?2	Hid2/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:
??2output/kernel
:?2output/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?	variables
?metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
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
<59"
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
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
*:(2Adam/ConA/kernel/m
:2Adam/ConA/bias/m
*:(2Adam/ConB/kernel/m
:2Adam/ConB/bias/m
+:)2Adam/ConA2/kernel/m
:2Adam/ConA2/bias/m
*:(	2Adam/ConC/kernel/m
:2Adam/ConC/bias/m
+:)(2Adam/ConA3/kernel/m
:(2Adam/ConA3/bias/m
+:)2Adam/ConB2/kernel/m
:2Adam/ConB2/bias/m
+:)((2Adam/ConA4/kernel/m
:(2Adam/ConA4/bias/m
+:)(2Adam/ConB3/kernel/m
:(2Adam/ConB3/bias/m
+:)2Adam/ConC2/kernel/m
:2Adam/ConC2/bias/m
*:(	2Adam/ConD/kernel/m
:2Adam/ConD/bias/m
*:( 2Adam/ConE/kernel/m
: 2Adam/ConE/bias/m
%:#???2Adam/HidA/kernel/m
:?2Adam/HidA/bias/m
&:$???2Adam/HidA2/kernel/m
:?2Adam/HidA2/bias/m
%:#
?F?2Adam/HidA3/kernel/m
:?2Adam/HidA3/bias/m
%:#
??2Adam/HidA4/kernel/m
:?2Adam/HidA4/bias/m
%:#ȓ?2Adam/HidB/kernel/m
:?2Adam/HidB/bias/m
&:$???2Adam/HidB2/kernel/m
:?2Adam/HidB2/bias/m
%:#
?F?2Adam/HidB3/kernel/m
:?2Adam/HidB3/bias/m
%:#???2Adam/HidC/kernel/m
:?2Adam/HidC/bias/m
&:$???2Adam/HidC2/kernel/m
:?2Adam/HidC2/bias/m
&:$???2Adam/HidD1/kernel/m
:?2Adam/HidD1/bias/m
&:$???2Adam/HidE1/kernel/m
:?2Adam/HidE1/bias/m
$:"
??2Adam/Hid1/kernel/m
:?2Adam/Hid1/bias/m
$:"
??2Adam/Hid2/kernel/m
:?2Adam/Hid2/bias/m
&:$
??2Adam/output/kernel/m
:?2Adam/output/bias/m
*:(2Adam/ConA/kernel/v
:2Adam/ConA/bias/v
*:(2Adam/ConB/kernel/v
:2Adam/ConB/bias/v
+:)2Adam/ConA2/kernel/v
:2Adam/ConA2/bias/v
*:(	2Adam/ConC/kernel/v
:2Adam/ConC/bias/v
+:)(2Adam/ConA3/kernel/v
:(2Adam/ConA3/bias/v
+:)2Adam/ConB2/kernel/v
:2Adam/ConB2/bias/v
+:)((2Adam/ConA4/kernel/v
:(2Adam/ConA4/bias/v
+:)(2Adam/ConB3/kernel/v
:(2Adam/ConB3/bias/v
+:)2Adam/ConC2/kernel/v
:2Adam/ConC2/bias/v
*:(	2Adam/ConD/kernel/v
:2Adam/ConD/bias/v
*:( 2Adam/ConE/kernel/v
: 2Adam/ConE/bias/v
%:#???2Adam/HidA/kernel/v
:?2Adam/HidA/bias/v
&:$???2Adam/HidA2/kernel/v
:?2Adam/HidA2/bias/v
%:#
?F?2Adam/HidA3/kernel/v
:?2Adam/HidA3/bias/v
%:#
??2Adam/HidA4/kernel/v
:?2Adam/HidA4/bias/v
%:#ȓ?2Adam/HidB/kernel/v
:?2Adam/HidB/bias/v
&:$???2Adam/HidB2/kernel/v
:?2Adam/HidB2/bias/v
%:#
?F?2Adam/HidB3/kernel/v
:?2Adam/HidB3/bias/v
%:#???2Adam/HidC/kernel/v
:?2Adam/HidC/bias/v
&:$???2Adam/HidC2/kernel/v
:?2Adam/HidC2/bias/v
&:$???2Adam/HidD1/kernel/v
:?2Adam/HidD1/bias/v
&:$???2Adam/HidE1/kernel/v
:?2Adam/HidE1/bias/v
$:"
??2Adam/Hid1/kernel/v
:?2Adam/Hid1/bias/v
$:"
??2Adam/Hid2/kernel/v
:?2Adam/Hid2/bias/v
&:$
??2Adam/output/kernel/v
:?2Adam/output/bias/v
?2?
A__inference_model_layer_call_and_return_conditional_losses_193597
A__inference_model_layer_call_and_return_conditional_losses_193899
A__inference_model_layer_call_and_return_conditional_losses_193096
A__inference_model_layer_call_and_return_conditional_losses_193259?
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
?B?
!__inference__wrapped_model_190788input"?
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
&__inference_model_layer_call_fn_191815
&__inference_model_layer_call_fn_194004
&__inference_model_layer_call_fn_194109
&__inference_model_layer_call_fn_192933?
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
@__inference_ConA_layer_call_and_return_conditional_losses_194120?
???
FullArgSpec
args?
jself
jinputs
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
%__inference_ConA_layer_call_fn_194129?
???
FullArgSpec
args?
jself
jinputs
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
A__inference_PoolA_layer_call_and_return_conditional_losses_194134
A__inference_PoolA_layer_call_and_return_conditional_losses_194139?
???
FullArgSpec
args?
jself
jinputs
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
&__inference_PoolA_layer_call_fn_194144
&__inference_PoolA_layer_call_fn_194149?
???
FullArgSpec
args?
jself
jinputs
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
C__inference_dropout_layer_call_and_return_conditional_losses_194154
C__inference_dropout_layer_call_and_return_conditional_losses_194166?
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
(__inference_dropout_layer_call_fn_194171
(__inference_dropout_layer_call_fn_194176?
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
@__inference_ConB_layer_call_and_return_conditional_losses_194187?
???
FullArgSpec
args?
jself
jinputs
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
%__inference_ConB_layer_call_fn_194196?
???
FullArgSpec
args?
jself
jinputs
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
A__inference_ConA2_layer_call_and_return_conditional_losses_194207?
???
FullArgSpec
args?
jself
jinputs
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
&__inference_ConA2_layer_call_fn_194216?
???
FullArgSpec
args?
jself
jinputs
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
A__inference_PoolB_layer_call_and_return_conditional_losses_194221
A__inference_PoolB_layer_call_and_return_conditional_losses_194226?
???
FullArgSpec
args?
jself
jinputs
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
&__inference_PoolB_layer_call_fn_194231
&__inference_PoolB_layer_call_fn_194236?
???
FullArgSpec
args?
jself
jinputs
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
B__inference_PoolA2_layer_call_and_return_conditional_losses_194241
B__inference_PoolA2_layer_call_and_return_conditional_losses_194246?
???
FullArgSpec
args?
jself
jinputs
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
'__inference_PoolA2_layer_call_fn_194251
'__inference_PoolA2_layer_call_fn_194256?
???
FullArgSpec
args?
jself
jinputs
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
E__inference_dropout_1_layer_call_and_return_conditional_losses_194261
E__inference_dropout_1_layer_call_and_return_conditional_losses_194273?
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
*__inference_dropout_1_layer_call_fn_194278
*__inference_dropout_1_layer_call_fn_194283?
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
@__inference_ConC_layer_call_and_return_conditional_losses_194294?
???
FullArgSpec
args?
jself
jinputs
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
%__inference_ConC_layer_call_fn_194303?
???
FullArgSpec
args?
jself
jinputs
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
A__inference_ConA3_layer_call_and_return_conditional_losses_194314?
???
FullArgSpec
args?
jself
jinputs
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
&__inference_ConA3_layer_call_fn_194323?
???
FullArgSpec
args?
jself
jinputs
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
A__inference_ConB2_layer_call_and_return_conditional_losses_194334?
???
FullArgSpec
args?
jself
jinputs
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
&__inference_ConB2_layer_call_fn_194343?
???
FullArgSpec
args?
jself
jinputs
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
A__inference_PoolC_layer_call_and_return_conditional_losses_194348
A__inference_PoolC_layer_call_and_return_conditional_losses_194353?
???
FullArgSpec
args?
jself
jinputs
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
&__inference_PoolC_layer_call_fn_194358
&__inference_PoolC_layer_call_fn_194363?
???
FullArgSpec
args?
jself
jinputs
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
B__inference_PoolA3_layer_call_and_return_conditional_losses_194368
B__inference_PoolA3_layer_call_and_return_conditional_losses_194373?
???
FullArgSpec
args?
jself
jinputs
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
'__inference_PoolA3_layer_call_fn_194378
'__inference_PoolA3_layer_call_fn_194383?
???
FullArgSpec
args?
jself
jinputs
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
B__inference_PoolB2_layer_call_and_return_conditional_losses_194388
B__inference_PoolB2_layer_call_and_return_conditional_losses_194393?
???
FullArgSpec
args?
jself
jinputs
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
'__inference_PoolB2_layer_call_fn_194398
'__inference_PoolB2_layer_call_fn_194403?
???
FullArgSpec
args?
jself
jinputs
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
E__inference_dropout_2_layer_call_and_return_conditional_losses_194408
E__inference_dropout_2_layer_call_and_return_conditional_losses_194420?
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
*__inference_dropout_2_layer_call_fn_194425
*__inference_dropout_2_layer_call_fn_194430?
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
A__inference_ConA4_layer_call_and_return_conditional_losses_194441?
???
FullArgSpec
args?
jself
jinputs
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
&__inference_ConA4_layer_call_fn_194450?
???
FullArgSpec
args?
jself
jinputs
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
A__inference_ConB3_layer_call_and_return_conditional_losses_194461?
???
FullArgSpec
args?
jself
jinputs
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
&__inference_ConB3_layer_call_fn_194470?
???
FullArgSpec
args?
jself
jinputs
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
A__inference_ConC2_layer_call_and_return_conditional_losses_194481?
???
FullArgSpec
args?
jself
jinputs
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
&__inference_ConC2_layer_call_fn_194490?
???
FullArgSpec
args?
jself
jinputs
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
@__inference_ConD_layer_call_and_return_conditional_losses_194501?
???
FullArgSpec
args?
jself
jinputs
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
%__inference_ConD_layer_call_fn_194510?
???
FullArgSpec
args?
jself
jinputs
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
@__inference_ConE_layer_call_and_return_conditional_losses_194521?
???
FullArgSpec
args?
jself
jinputs
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
%__inference_ConE_layer_call_fn_194530?
???
FullArgSpec
args?
jself
jinputs
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
B__inference_PoolA4_layer_call_and_return_conditional_losses_194535
B__inference_PoolA4_layer_call_and_return_conditional_losses_194540?
???
FullArgSpec
args?
jself
jinputs
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
'__inference_PoolA4_layer_call_fn_194545
'__inference_PoolA4_layer_call_fn_194550?
???
FullArgSpec
args?
jself
jinputs
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
B__inference_PoolB3_layer_call_and_return_conditional_losses_194555
B__inference_PoolB3_layer_call_and_return_conditional_losses_194560?
???
FullArgSpec
args?
jself
jinputs
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
'__inference_PoolB3_layer_call_fn_194565
'__inference_PoolB3_layer_call_fn_194570?
???
FullArgSpec
args?
jself
jinputs
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
B__inference_PoolC2_layer_call_and_return_conditional_losses_194575
B__inference_PoolC2_layer_call_and_return_conditional_losses_194580?
???
FullArgSpec
args?
jself
jinputs
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
'__inference_PoolC2_layer_call_fn_194585
'__inference_PoolC2_layer_call_fn_194590?
???
FullArgSpec
args?
jself
jinputs
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
A__inference_PoolD_layer_call_and_return_conditional_losses_194595
A__inference_PoolD_layer_call_and_return_conditional_losses_194600?
???
FullArgSpec
args?
jself
jinputs
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
&__inference_PoolD_layer_call_fn_194605
&__inference_PoolD_layer_call_fn_194610?
???
FullArgSpec
args?
jself
jinputs
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
A__inference_PoolE_layer_call_and_return_conditional_losses_194615
A__inference_PoolE_layer_call_and_return_conditional_losses_194620?
???
FullArgSpec
args?
jself
jinputs
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
&__inference_PoolE_layer_call_fn_194625
&__inference_PoolE_layer_call_fn_194630?
???
FullArgSpec
args?
jself
jinputs
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
E__inference_dropout_5_layer_call_and_return_conditional_losses_194635
E__inference_dropout_5_layer_call_and_return_conditional_losses_194647?
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
*__inference_dropout_5_layer_call_fn_194652
*__inference_dropout_5_layer_call_fn_194657?
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
E__inference_dropout_6_layer_call_and_return_conditional_losses_194662
E__inference_dropout_6_layer_call_and_return_conditional_losses_194674?
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
*__inference_dropout_6_layer_call_fn_194679
*__inference_dropout_6_layer_call_fn_194684?
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
E__inference_dropout_7_layer_call_and_return_conditional_losses_194689
E__inference_dropout_7_layer_call_and_return_conditional_losses_194701?
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
*__inference_dropout_7_layer_call_fn_194706
*__inference_dropout_7_layer_call_fn_194711?
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
E__inference_dropout_8_layer_call_and_return_conditional_losses_194716
E__inference_dropout_8_layer_call_and_return_conditional_losses_194728?
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
*__inference_dropout_8_layer_call_fn_194733
*__inference_dropout_8_layer_call_fn_194738?
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
E__inference_dropout_9_layer_call_and_return_conditional_losses_194743
E__inference_dropout_9_layer_call_and_return_conditional_losses_194755?
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
*__inference_dropout_9_layer_call_fn_194760
*__inference_dropout_9_layer_call_fn_194765?
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
F__inference_dropout_10_layer_call_and_return_conditional_losses_194770
F__inference_dropout_10_layer_call_and_return_conditional_losses_194782?
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
+__inference_dropout_10_layer_call_fn_194787
+__inference_dropout_10_layer_call_fn_194792?
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
E__inference_dropout_3_layer_call_and_return_conditional_losses_194797
E__inference_dropout_3_layer_call_and_return_conditional_losses_194809?
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
*__inference_dropout_3_layer_call_fn_194814
*__inference_dropout_3_layer_call_fn_194819?
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
E__inference_dropout_4_layer_call_and_return_conditional_losses_194824
E__inference_dropout_4_layer_call_and_return_conditional_losses_194836?
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
*__inference_dropout_4_layer_call_fn_194841
*__inference_dropout_4_layer_call_fn_194846?
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
C__inference_flatten_layer_call_and_return_conditional_losses_194852?
???
FullArgSpec
args?
jself
jinputs
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
(__inference_flatten_layer_call_fn_194857?
???
FullArgSpec
args?
jself
jinputs
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
E__inference_flatten_1_layer_call_and_return_conditional_losses_194863?
???
FullArgSpec
args?
jself
jinputs
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
*__inference_flatten_1_layer_call_fn_194868?
???
FullArgSpec
args?
jself
jinputs
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
E__inference_flatten_2_layer_call_and_return_conditional_losses_194874?
???
FullArgSpec
args?
jself
jinputs
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
*__inference_flatten_2_layer_call_fn_194879?
???
FullArgSpec
args?
jself
jinputs
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
E__inference_flatten_3_layer_call_and_return_conditional_losses_194885?
???
FullArgSpec
args?
jself
jinputs
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
*__inference_flatten_3_layer_call_fn_194890?
???
FullArgSpec
args?
jself
jinputs
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
E__inference_flatten_4_layer_call_and_return_conditional_losses_194896?
???
FullArgSpec
args?
jself
jinputs
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
*__inference_flatten_4_layer_call_fn_194901?
???
FullArgSpec
args?
jself
jinputs
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
E__inference_flatten_5_layer_call_and_return_conditional_losses_194907?
???
FullArgSpec
args?
jself
jinputs
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
*__inference_flatten_5_layer_call_fn_194912?
???
FullArgSpec
args?
jself
jinputs
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
E__inference_flatten_6_layer_call_and_return_conditional_losses_194918?
???
FullArgSpec
args?
jself
jinputs
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
*__inference_flatten_6_layer_call_fn_194923?
???
FullArgSpec
args?
jself
jinputs
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
E__inference_flatten_7_layer_call_and_return_conditional_losses_194929?
???
FullArgSpec
args?
jself
jinputs
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
*__inference_flatten_7_layer_call_fn_194934?
???
FullArgSpec
args?
jself
jinputs
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
E__inference_flatten_8_layer_call_and_return_conditional_losses_194940?
???
FullArgSpec
args?
jself
jinputs
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
*__inference_flatten_8_layer_call_fn_194945?
???
FullArgSpec
args?
jself
jinputs
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
E__inference_flatten_9_layer_call_and_return_conditional_losses_194951?
???
FullArgSpec
args?
jself
jinputs
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
*__inference_flatten_9_layer_call_fn_194956?
???
FullArgSpec
args?
jself
jinputs
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
F__inference_flatten_10_layer_call_and_return_conditional_losses_194962?
???
FullArgSpec
args?
jself
jinputs
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
+__inference_flatten_10_layer_call_fn_194967?
???
FullArgSpec
args?
jself
jinputs
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
@__inference_HidA_layer_call_and_return_conditional_losses_194978?
???
FullArgSpec
args?
jself
jinputs
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
%__inference_HidA_layer_call_fn_194987?
???
FullArgSpec
args?
jself
jinputs
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
A__inference_HidA2_layer_call_and_return_conditional_losses_194998?
???
FullArgSpec
args?
jself
jinputs
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
&__inference_HidA2_layer_call_fn_195007?
???
FullArgSpec
args?
jself
jinputs
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
A__inference_HidA3_layer_call_and_return_conditional_losses_195018?
???
FullArgSpec
args?
jself
jinputs
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
&__inference_HidA3_layer_call_fn_195027?
???
FullArgSpec
args?
jself
jinputs
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
A__inference_HidA4_layer_call_and_return_conditional_losses_195038?
???
FullArgSpec
args?
jself
jinputs
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
&__inference_HidA4_layer_call_fn_195047?
???
FullArgSpec
args?
jself
jinputs
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
@__inference_HidB_layer_call_and_return_conditional_losses_195058?
???
FullArgSpec
args?
jself
jinputs
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
%__inference_HidB_layer_call_fn_195067?
???
FullArgSpec
args?
jself
jinputs
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
A__inference_HidB2_layer_call_and_return_conditional_losses_195078?
???
FullArgSpec
args?
jself
jinputs
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
&__inference_HidB2_layer_call_fn_195087?
???
FullArgSpec
args?
jself
jinputs
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
A__inference_HidB3_layer_call_and_return_conditional_losses_195098?
???
FullArgSpec
args?
jself
jinputs
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
&__inference_HidB3_layer_call_fn_195107?
???
FullArgSpec
args?
jself
jinputs
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
@__inference_HidC_layer_call_and_return_conditional_losses_195118?
???
FullArgSpec
args?
jself
jinputs
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
%__inference_HidC_layer_call_fn_195127?
???
FullArgSpec
args?
jself
jinputs
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
A__inference_HidC2_layer_call_and_return_conditional_losses_195138?
???
FullArgSpec
args?
jself
jinputs
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
&__inference_HidC2_layer_call_fn_195147?
???
FullArgSpec
args?
jself
jinputs
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
A__inference_HidD1_layer_call_and_return_conditional_losses_195158?
???
FullArgSpec
args?
jself
jinputs
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
&__inference_HidD1_layer_call_fn_195167?
???
FullArgSpec
args?
jself
jinputs
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
A__inference_HidE1_layer_call_and_return_conditional_losses_195178?
???
FullArgSpec
args?
jself
jinputs
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
&__inference_HidE1_layer_call_fn_195187?
???
FullArgSpec
args?
jself
jinputs
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
G__inference_concatenate_layer_call_and_return_conditional_losses_195203?
???
FullArgSpec
args?
jself
jinputs
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
,__inference_concatenate_layer_call_fn_195218?
???
FullArgSpec
args?
jself
jinputs
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
@__inference_Hid1_layer_call_and_return_conditional_losses_195229?
???
FullArgSpec
args?
jself
jinputs
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
%__inference_Hid1_layer_call_fn_195238?
???
FullArgSpec
args?
jself
jinputs
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
@__inference_Hid2_layer_call_and_return_conditional_losses_195249?
???
FullArgSpec
args?
jself
jinputs
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
%__inference_Hid2_layer_call_fn_195258?
???
FullArgSpec
args?
jself
jinputs
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
B__inference_output_layer_call_and_return_conditional_losses_195269?
???
FullArgSpec
args?
jself
jinputs
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
'__inference_output_layer_call_fn_195278?
???
FullArgSpec
args?
jself
jinputs
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
$__inference_signature_wrapper_193372input"?
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
A__inference_ConA2_layer_call_and_return_conditional_losses_194207lWX7?4
-?*
(?%
inputs?????????I?
? "-?*
#? 
0?????????G=
? ?
&__inference_ConA2_layer_call_fn_194216_WX7?4
-?*
(?%
inputs?????????I?
? " ??????????G=?
A__inference_ConA3_layer_call_and_return_conditional_losses_194314lop7?4
-?*
(?%
inputs?????????#
? "-?*
#? 
0?????????!(
? ?
&__inference_ConA3_layer_call_fn_194323_op7?4
-?*
(?%
inputs?????????#
? " ??????????!(?
A__inference_ConA4_layer_call_and_return_conditional_losses_194441n??7?4
-?*
(?%
inputs?????????(
? "-?*
#? 
0?????????(
? ?
&__inference_ConA4_layer_call_fn_194450a??7?4
-?*
(?%
inputs?????????(
? " ??????????(?
@__inference_ConA_layer_call_and_return_conditional_losses_194120pCD9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
%__inference_ConA_layer_call_fn_194129cCD9?6
/?,
*?'
inputs???????????
? ""?????????????
A__inference_ConB2_layer_call_and_return_conditional_losses_194334luv7?4
-?*
(?%
inputs?????????H?
? "-?*
#? 
0?????????F=
? ?
&__inference_ConB2_layer_call_fn_194343_uv7?4
-?*
(?%
inputs?????????H?
? " ??????????F=?
A__inference_ConB3_layer_call_and_return_conditional_losses_194461n??7?4
-?*
(?%
inputs?????????#
? "-?*
#? 
0?????????!(
? ?
&__inference_ConB3_layer_call_fn_194470a??7?4
-?*
(?%
inputs?????????#
? " ??????????!(?
@__inference_ConB_layer_call_and_return_conditional_losses_194187pQR9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
%__inference_ConB_layer_call_fn_194196cQR9?6
/?,
*?'
inputs???????????
? ""?????????????
A__inference_ConC2_layer_call_and_return_conditional_losses_194481n??7?4
-?*
(?%
inputs?????????G>
? "-?*
#? 
0?????????E<
? ?
&__inference_ConC2_layer_call_fn_194490a??7?4
-?*
(?%
inputs?????????G>
? " ??????????E<?
@__inference_ConC_layer_call_and_return_conditional_losses_194294pij9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
%__inference_ConC_layer_call_fn_194303cij9?6
/?,
*?'
inputs???????????
? ""?????????????
@__inference_ConD_layer_call_and_return_conditional_losses_194501r??9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
%__inference_ConD_layer_call_fn_194510e??9?6
/?,
*?'
inputs???????????
? ""?????????????
@__inference_ConE_layer_call_and_return_conditional_losses_194521r??9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0??????????? 
? ?
%__inference_ConE_layer_call_fn_194530e??9?6
/?,
*?'
inputs???????????
? ""???????????? ?
@__inference_Hid1_layer_call_and_return_conditional_losses_195229`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? |
%__inference_Hid1_layer_call_fn_195238S??0?-
&?#
!?
inputs??????????
? "????????????
@__inference_Hid2_layer_call_and_return_conditional_losses_195249`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? |
%__inference_Hid2_layer_call_fn_195258S??0?-
&?#
!?
inputs??????????
? "????????????
A__inference_HidA2_layer_call_and_return_conditional_losses_194998a??1?.
'?$
"?
inputs???????????
? "&?#
?
0??????????
? ~
&__inference_HidA2_layer_call_fn_195007T??1?.
'?$
"?
inputs???????????
? "????????????
A__inference_HidA3_layer_call_and_return_conditional_losses_195018`??0?-
&?#
!?
inputs??????????F
? "&?#
?
0??????????
? }
&__inference_HidA3_layer_call_fn_195027S??0?-
&?#
!?
inputs??????????F
? "????????????
A__inference_HidA4_layer_call_and_return_conditional_losses_195038`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
&__inference_HidA4_layer_call_fn_195047S??0?-
&?#
!?
inputs??????????
? "????????????
@__inference_HidA_layer_call_and_return_conditional_losses_194978a??1?.
'?$
"?
inputs???????????
? "&?#
?
0??????????
? }
%__inference_HidA_layer_call_fn_194987T??1?.
'?$
"?
inputs???????????
? "????????????
A__inference_HidB2_layer_call_and_return_conditional_losses_195078a??1?.
'?$
"?
inputs???????????
? "&?#
?
0??????????
? ~
&__inference_HidB2_layer_call_fn_195087T??1?.
'?$
"?
inputs???????????
? "????????????
A__inference_HidB3_layer_call_and_return_conditional_losses_195098`??0?-
&?#
!?
inputs??????????F
? "&?#
?
0??????????
? }
&__inference_HidB3_layer_call_fn_195107S??0?-
&?#
!?
inputs??????????F
? "????????????
@__inference_HidB_layer_call_and_return_conditional_losses_195058a??1?.
'?$
"?
inputs?????????ȓ
? "&?#
?
0??????????
? }
%__inference_HidB_layer_call_fn_195067T??1?.
'?$
"?
inputs?????????ȓ
? "????????????
A__inference_HidC2_layer_call_and_return_conditional_losses_195138a??1?.
'?$
"?
inputs???????????
? "&?#
?
0??????????
? ~
&__inference_HidC2_layer_call_fn_195147T??1?.
'?$
"?
inputs???????????
? "????????????
@__inference_HidC_layer_call_and_return_conditional_losses_195118a??1?.
'?$
"?
inputs???????????
? "&?#
?
0??????????
? }
%__inference_HidC_layer_call_fn_195127T??1?.
'?$
"?
inputs???????????
? "????????????
A__inference_HidD1_layer_call_and_return_conditional_losses_195158a??1?.
'?$
"?
inputs???????????
? "&?#
?
0??????????
? ~
&__inference_HidD1_layer_call_fn_195167T??1?.
'?$
"?
inputs???????????
? "????????????
A__inference_HidE1_layer_call_and_return_conditional_losses_195178a??1?.
'?$
"?
inputs???????????
? "&?#
?
0??????????
? ~
&__inference_HidE1_layer_call_fn_195187T??1?.
'?$
"?
inputs???????????
? "????????????
B__inference_PoolA2_layer_call_and_return_conditional_losses_194241?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
B__inference_PoolA2_layer_call_and_return_conditional_losses_194246h7?4
-?*
(?%
inputs?????????G=
? "-?*
#? 
0?????????#
? ?
'__inference_PoolA2_layer_call_fn_194251?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
'__inference_PoolA2_layer_call_fn_194256[7?4
-?*
(?%
inputs?????????G=
? " ??????????#?
B__inference_PoolA3_layer_call_and_return_conditional_losses_194368?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
B__inference_PoolA3_layer_call_and_return_conditional_losses_194373h7?4
-?*
(?%
inputs?????????!(
? "-?*
#? 
0?????????(
? ?
'__inference_PoolA3_layer_call_fn_194378?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
'__inference_PoolA3_layer_call_fn_194383[7?4
-?*
(?%
inputs?????????!(
? " ??????????(?
B__inference_PoolA4_layer_call_and_return_conditional_losses_194535?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
B__inference_PoolA4_layer_call_and_return_conditional_losses_194540h7?4
-?*
(?%
inputs?????????(
? "-?*
#? 
0?????????(
? ?
'__inference_PoolA4_layer_call_fn_194545?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
'__inference_PoolA4_layer_call_fn_194550[7?4
-?*
(?%
inputs?????????(
? " ??????????(?
A__inference_PoolA_layer_call_and_return_conditional_losses_194134?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
A__inference_PoolA_layer_call_and_return_conditional_losses_194139j9?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????I?
? ?
&__inference_PoolA_layer_call_fn_194144?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
&__inference_PoolA_layer_call_fn_194149]9?6
/?,
*?'
inputs???????????
? " ??????????I??
B__inference_PoolB2_layer_call_and_return_conditional_losses_194388?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
B__inference_PoolB2_layer_call_and_return_conditional_losses_194393h7?4
-?*
(?%
inputs?????????F=
? "-?*
#? 
0?????????#
? ?
'__inference_PoolB2_layer_call_fn_194398?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
'__inference_PoolB2_layer_call_fn_194403[7?4
-?*
(?%
inputs?????????F=
? " ??????????#?
B__inference_PoolB3_layer_call_and_return_conditional_losses_194555?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
B__inference_PoolB3_layer_call_and_return_conditional_losses_194560h7?4
-?*
(?%
inputs?????????!(
? "-?*
#? 
0?????????(
? ?
'__inference_PoolB3_layer_call_fn_194565?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
'__inference_PoolB3_layer_call_fn_194570[7?4
-?*
(?%
inputs?????????!(
? " ??????????(?
A__inference_PoolB_layer_call_and_return_conditional_losses_194221?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
A__inference_PoolB_layer_call_and_return_conditional_losses_194226j9?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????H?
? ?
&__inference_PoolB_layer_call_fn_194231?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
&__inference_PoolB_layer_call_fn_194236]9?6
/?,
*?'
inputs???????????
? " ??????????H??
B__inference_PoolC2_layer_call_and_return_conditional_losses_194575?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
B__inference_PoolC2_layer_call_and_return_conditional_losses_194580h7?4
-?*
(?%
inputs?????????E<
? "-?*
#? 
0?????????"
? ?
'__inference_PoolC2_layer_call_fn_194585?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
'__inference_PoolC2_layer_call_fn_194590[7?4
-?*
(?%
inputs?????????E<
? " ??????????"?
A__inference_PoolC_layer_call_and_return_conditional_losses_194348?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
A__inference_PoolC_layer_call_and_return_conditional_losses_194353j9?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????G>
? ?
&__inference_PoolC_layer_call_fn_194358?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
&__inference_PoolC_layer_call_fn_194363]9?6
/?,
*?'
inputs???????????
? " ??????????G>?
A__inference_PoolD_layer_call_and_return_conditional_losses_194595?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
A__inference_PoolD_layer_call_and_return_conditional_losses_194600j9?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????G=
? ?
&__inference_PoolD_layer_call_fn_194605?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
&__inference_PoolD_layer_call_fn_194610]9?6
/?,
*?'
inputs???????????
? " ??????????G=?
A__inference_PoolE_layer_call_and_return_conditional_losses_194615?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
A__inference_PoolE_layer_call_and_return_conditional_losses_194620j9?6
/?,
*?'
inputs??????????? 
? "-?*
#? 
0?????????D; 
? ?
&__inference_PoolE_layer_call_fn_194625?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
&__inference_PoolE_layer_call_fn_194630]9?6
/?,
*?'
inputs??????????? 
? " ??????????D; ?
!__inference__wrapped_model_190788?XCDQRWXijuvop??????????????????????????????????????8?5
.?+
)?&
input???????????
? "0?-
+
output!?
output???????????
G__inference_concatenate_layer_call_and_return_conditional_losses_195203????
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
? "&?#
?
0??????????
? ?
,__inference_concatenate_layer_call_fn_195218????
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
? "????????????
F__inference_dropout_10_layer_call_and_return_conditional_losses_194770l;?8
1?.
(?%
inputs?????????"
p 
? "-?*
#? 
0?????????"
? ?
F__inference_dropout_10_layer_call_and_return_conditional_losses_194782l;?8
1?.
(?%
inputs?????????"
p
? "-?*
#? 
0?????????"
? ?
+__inference_dropout_10_layer_call_fn_194787_;?8
1?.
(?%
inputs?????????"
p 
? " ??????????"?
+__inference_dropout_10_layer_call_fn_194792_;?8
1?.
(?%
inputs?????????"
p
? " ??????????"?
E__inference_dropout_1_layer_call_and_return_conditional_losses_194261l;?8
1?.
(?%
inputs?????????H?
p 
? "-?*
#? 
0?????????H?
? ?
E__inference_dropout_1_layer_call_and_return_conditional_losses_194273l;?8
1?.
(?%
inputs?????????H?
p
? "-?*
#? 
0?????????H?
? ?
*__inference_dropout_1_layer_call_fn_194278_;?8
1?.
(?%
inputs?????????H?
p 
? " ??????????H??
*__inference_dropout_1_layer_call_fn_194283_;?8
1?.
(?%
inputs?????????H?
p
? " ??????????H??
E__inference_dropout_2_layer_call_and_return_conditional_losses_194408l;?8
1?.
(?%
inputs?????????G>
p 
? "-?*
#? 
0?????????G>
? ?
E__inference_dropout_2_layer_call_and_return_conditional_losses_194420l;?8
1?.
(?%
inputs?????????G>
p
? "-?*
#? 
0?????????G>
? ?
*__inference_dropout_2_layer_call_fn_194425_;?8
1?.
(?%
inputs?????????G>
p 
? " ??????????G>?
*__inference_dropout_2_layer_call_fn_194430_;?8
1?.
(?%
inputs?????????G>
p
? " ??????????G>?
E__inference_dropout_3_layer_call_and_return_conditional_losses_194797l;?8
1?.
(?%
inputs?????????G=
p 
? "-?*
#? 
0?????????G=
? ?
E__inference_dropout_3_layer_call_and_return_conditional_losses_194809l;?8
1?.
(?%
inputs?????????G=
p
? "-?*
#? 
0?????????G=
? ?
*__inference_dropout_3_layer_call_fn_194814_;?8
1?.
(?%
inputs?????????G=
p 
? " ??????????G=?
*__inference_dropout_3_layer_call_fn_194819_;?8
1?.
(?%
inputs?????????G=
p
? " ??????????G=?
E__inference_dropout_4_layer_call_and_return_conditional_losses_194824l;?8
1?.
(?%
inputs?????????D; 
p 
? "-?*
#? 
0?????????D; 
? ?
E__inference_dropout_4_layer_call_and_return_conditional_losses_194836l;?8
1?.
(?%
inputs?????????D; 
p
? "-?*
#? 
0?????????D; 
? ?
*__inference_dropout_4_layer_call_fn_194841_;?8
1?.
(?%
inputs?????????D; 
p 
? " ??????????D; ?
*__inference_dropout_4_layer_call_fn_194846_;?8
1?.
(?%
inputs?????????D; 
p
? " ??????????D; ?
E__inference_dropout_5_layer_call_and_return_conditional_losses_194635l;?8
1?.
(?%
inputs?????????#
p 
? "-?*
#? 
0?????????#
? ?
E__inference_dropout_5_layer_call_and_return_conditional_losses_194647l;?8
1?.
(?%
inputs?????????#
p
? "-?*
#? 
0?????????#
? ?
*__inference_dropout_5_layer_call_fn_194652_;?8
1?.
(?%
inputs?????????#
p 
? " ??????????#?
*__inference_dropout_5_layer_call_fn_194657_;?8
1?.
(?%
inputs?????????#
p
? " ??????????#?
E__inference_dropout_6_layer_call_and_return_conditional_losses_194662l;?8
1?.
(?%
inputs?????????(
p 
? "-?*
#? 
0?????????(
? ?
E__inference_dropout_6_layer_call_and_return_conditional_losses_194674l;?8
1?.
(?%
inputs?????????(
p
? "-?*
#? 
0?????????(
? ?
*__inference_dropout_6_layer_call_fn_194679_;?8
1?.
(?%
inputs?????????(
p 
? " ??????????(?
*__inference_dropout_6_layer_call_fn_194684_;?8
1?.
(?%
inputs?????????(
p
? " ??????????(?
E__inference_dropout_7_layer_call_and_return_conditional_losses_194689l;?8
1?.
(?%
inputs?????????(
p 
? "-?*
#? 
0?????????(
? ?
E__inference_dropout_7_layer_call_and_return_conditional_losses_194701l;?8
1?.
(?%
inputs?????????(
p
? "-?*
#? 
0?????????(
? ?
*__inference_dropout_7_layer_call_fn_194706_;?8
1?.
(?%
inputs?????????(
p 
? " ??????????(?
*__inference_dropout_7_layer_call_fn_194711_;?8
1?.
(?%
inputs?????????(
p
? " ??????????(?
E__inference_dropout_8_layer_call_and_return_conditional_losses_194716l;?8
1?.
(?%
inputs?????????#
p 
? "-?*
#? 
0?????????#
? ?
E__inference_dropout_8_layer_call_and_return_conditional_losses_194728l;?8
1?.
(?%
inputs?????????#
p
? "-?*
#? 
0?????????#
? ?
*__inference_dropout_8_layer_call_fn_194733_;?8
1?.
(?%
inputs?????????#
p 
? " ??????????#?
*__inference_dropout_8_layer_call_fn_194738_;?8
1?.
(?%
inputs?????????#
p
? " ??????????#?
E__inference_dropout_9_layer_call_and_return_conditional_losses_194743l;?8
1?.
(?%
inputs?????????(
p 
? "-?*
#? 
0?????????(
? ?
E__inference_dropout_9_layer_call_and_return_conditional_losses_194755l;?8
1?.
(?%
inputs?????????(
p
? "-?*
#? 
0?????????(
? ?
*__inference_dropout_9_layer_call_fn_194760_;?8
1?.
(?%
inputs?????????(
p 
? " ??????????(?
*__inference_dropout_9_layer_call_fn_194765_;?8
1?.
(?%
inputs?????????(
p
? " ??????????(?
C__inference_dropout_layer_call_and_return_conditional_losses_194154l;?8
1?.
(?%
inputs?????????I?
p 
? "-?*
#? 
0?????????I?
? ?
C__inference_dropout_layer_call_and_return_conditional_losses_194166l;?8
1?.
(?%
inputs?????????I?
p
? "-?*
#? 
0?????????I?
? ?
(__inference_dropout_layer_call_fn_194171_;?8
1?.
(?%
inputs?????????I?
p 
? " ??????????I??
(__inference_dropout_layer_call_fn_194176_;?8
1?.
(?%
inputs?????????I?
p
? " ??????????I??
F__inference_flatten_10_layer_call_and_return_conditional_losses_194962b7?4
-?*
(?%
inputs?????????D; 
? "'?$
?
0???????????
? ?
+__inference_flatten_10_layer_call_fn_194967U7?4
-?*
(?%
inputs?????????D; 
? "?????????????
E__inference_flatten_1_layer_call_and_return_conditional_losses_194863b7?4
-?*
(?%
inputs?????????#
? "'?$
?
0???????????
? ?
*__inference_flatten_1_layer_call_fn_194868U7?4
-?*
(?%
inputs?????????#
? "?????????????
E__inference_flatten_2_layer_call_and_return_conditional_losses_194874a7?4
-?*
(?%
inputs?????????(
? "&?#
?
0??????????F
? ?
*__inference_flatten_2_layer_call_fn_194879T7?4
-?*
(?%
inputs?????????(
? "???????????F?
E__inference_flatten_3_layer_call_and_return_conditional_losses_194885a7?4
-?*
(?%
inputs?????????(
? "&?#
?
0??????????
? ?
*__inference_flatten_3_layer_call_fn_194890T7?4
-?*
(?%
inputs?????????(
? "????????????
E__inference_flatten_4_layer_call_and_return_conditional_losses_194896b7?4
-?*
(?%
inputs?????????H?
? "'?$
?
0?????????ȓ
? ?
*__inference_flatten_4_layer_call_fn_194901U7?4
-?*
(?%
inputs?????????H?
? "??????????ȓ?
E__inference_flatten_5_layer_call_and_return_conditional_losses_194907b7?4
-?*
(?%
inputs?????????#
? "'?$
?
0???????????
? ?
*__inference_flatten_5_layer_call_fn_194912U7?4
-?*
(?%
inputs?????????#
? "?????????????
E__inference_flatten_6_layer_call_and_return_conditional_losses_194918a7?4
-?*
(?%
inputs?????????(
? "&?#
?
0??????????F
? ?
*__inference_flatten_6_layer_call_fn_194923T7?4
-?*
(?%
inputs?????????(
? "???????????F?
E__inference_flatten_7_layer_call_and_return_conditional_losses_194929b7?4
-?*
(?%
inputs?????????G>
? "'?$
?
0???????????
? ?
*__inference_flatten_7_layer_call_fn_194934U7?4
-?*
(?%
inputs?????????G>
? "?????????????
E__inference_flatten_8_layer_call_and_return_conditional_losses_194940b7?4
-?*
(?%
inputs?????????"
? "'?$
?
0???????????
? ?
*__inference_flatten_8_layer_call_fn_194945U7?4
-?*
(?%
inputs?????????"
? "?????????????
E__inference_flatten_9_layer_call_and_return_conditional_losses_194951b7?4
-?*
(?%
inputs?????????G=
? "'?$
?
0???????????
? ?
*__inference_flatten_9_layer_call_fn_194956U7?4
-?*
(?%
inputs?????????G=
? "?????????????
C__inference_flatten_layer_call_and_return_conditional_losses_194852b7?4
-?*
(?%
inputs?????????I?
? "'?$
?
0???????????
? ?
(__inference_flatten_layer_call_fn_194857U7?4
-?*
(?%
inputs?????????I?
? "?????????????
A__inference_model_layer_call_and_return_conditional_losses_193096?XCDQRWXijuvop??????????????????????????????????????@?=
6?3
)?&
input???????????
p 

 
? "&?#
?
0??????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_193259?XCDQRWXijuvop??????????????????????????????????????@?=
6?3
)?&
input???????????
p

 
? "&?#
?
0??????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_193597?XCDQRWXijuvop??????????????????????????????????????A?>
7?4
*?'
inputs???????????
p 

 
? "&?#
?
0??????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_193899?XCDQRWXijuvop??????????????????????????????????????A?>
7?4
*?'
inputs???????????
p

 
? "&?#
?
0??????????
? ?
&__inference_model_layer_call_fn_191815?XCDQRWXijuvop??????????????????????????????????????@?=
6?3
)?&
input???????????
p 

 
? "????????????
&__inference_model_layer_call_fn_192933?XCDQRWXijuvop??????????????????????????????????????@?=
6?3
)?&
input???????????
p

 
? "????????????
&__inference_model_layer_call_fn_194004?XCDQRWXijuvop??????????????????????????????????????A?>
7?4
*?'
inputs???????????
p 

 
? "????????????
&__inference_model_layer_call_fn_194109?XCDQRWXijuvop??????????????????????????????????????A?>
7?4
*?'
inputs???????????
p

 
? "????????????
B__inference_output_layer_call_and_return_conditional_losses_195269`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
'__inference_output_layer_call_fn_195278S??0?-
&?#
!?
inputs??????????
? "????????????
$__inference_signature_wrapper_193372?XCDQRWXijuvop??????????????????????????????????????A?>
? 
7?4
2
input)?&
input???????????"0?-
+
output!?
output??????????