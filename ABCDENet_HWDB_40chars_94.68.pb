/
×
B
AssignVariableOp
resource
value"dtype"
dtypetype
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

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

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
delete_old_dirsbool(
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
¾
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
executor_typestring 
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.22v2.6.1-9-gc2363d6d0258û(
z
ConA/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConA/kernel
s
ConA/kernel/Read/ReadVariableOpReadVariableOpConA/kernel*&
_output_shapes
:*
dtype0
j
	ConA/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	ConA/bias
c
ConA/bias/Read/ReadVariableOpReadVariableOp	ConA/bias*
_output_shapes
:*
dtype0
z
ConB/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConB/kernel
s
ConB/kernel/Read/ReadVariableOpReadVariableOpConB/kernel*&
_output_shapes
:*
dtype0
j
	ConB/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	ConB/bias
c
ConB/bias/Read/ReadVariableOpReadVariableOp	ConB/bias*
_output_shapes
:*
dtype0
z
ConC/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameConC/kernel
s
ConC/kernel/Read/ReadVariableOpReadVariableOpConC/kernel*&
_output_shapes
:	*
dtype0
j
	ConC/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	ConC/bias
c
ConC/bias/Read/ReadVariableOpReadVariableOp	ConC/bias*
_output_shapes
:*
dtype0
|
ConA2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConA2/kernel
u
 ConA2/kernel/Read/ReadVariableOpReadVariableOpConA2/kernel*&
_output_shapes
:*
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
ConB2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConB2/kernel
u
 ConB2/kernel/Read/ReadVariableOpReadVariableOpConB2/kernel*&
_output_shapes
:*
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
shape:*
shared_nameConC2/kernel
u
 ConC2/kernel/Read/ReadVariableOpReadVariableOpConC2/kernel*&
_output_shapes
:*
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
shape:	*
shared_nameConD/kernel
s
ConD/kernel/Read/ReadVariableOpReadVariableOpConD/kernel*&
_output_shapes
:	*
dtype0
j
	ConD/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	ConD/bias
c
ConD/bias/Read/ReadVariableOpReadVariableOp	ConD/bias*
_output_shapes
:*
dtype0
z
ConE/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConE/kernel
s
ConE/kernel/Read/ReadVariableOpReadVariableOpConE/kernel*&
_output_shapes
:*
dtype0
j
	ConE/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	ConE/bias
c
ConE/bias/Read/ReadVariableOpReadVariableOp	ConE/bias*
_output_shapes
:*
dtype0
u
HidA/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ê×*
shared_nameHidA/kernel
n
HidA/kernel/Read/ReadVariableOpReadVariableOpHidA/kernel*!
_output_shapes
:Ê×*
dtype0
k
	HidA/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	HidA/bias
d
HidA/bias/Read/ReadVariableOpReadVariableOp	HidA/bias*
_output_shapes	
:*
dtype0
w
HidA2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:¤*
shared_nameHidA2/kernel
p
 HidA2/kernel/Read/ReadVariableOpReadVariableOpHidA2/kernel*!
_output_shapes
:¤*
dtype0
m

HidA2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
HidA2/bias
f
HidA2/bias/Read/ReadVariableOpReadVariableOp
HidA2/bias*
_output_shapes	
:*
dtype0
v
HidA3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
F*
shared_nameHidA3/kernel
o
 HidA3/kernel/Read/ReadVariableOpReadVariableOpHidA3/kernel* 
_output_shapes
:
F*
dtype0
m

HidA3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
HidA3/bias
f
HidA3/bias/Read/ReadVariableOpReadVariableOp
HidA3/bias*
_output_shapes	
:*
dtype0
u
HidB/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: ©*
shared_nameHidB/kernel
n
HidB/kernel/Read/ReadVariableOpReadVariableOpHidB/kernel*!
_output_shapes
: ©*
dtype0
k
	HidB/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	HidB/bias
d
HidB/bias/Read/ReadVariableOpReadVariableOp	HidB/bias*
_output_shapes	
:*
dtype0
w
HidB2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:¤*
shared_nameHidB2/kernel
p
 HidB2/kernel/Read/ReadVariableOpReadVariableOpHidB2/kernel*!
_output_shapes
:¤*
dtype0
m

HidB2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
HidB2/bias
f
HidB2/bias/Read/ReadVariableOpReadVariableOp
HidB2/bias*
_output_shapes	
:*
dtype0
v
HidB3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
F*
shared_nameHidB3/kernel
o
 HidB3/kernel/Read/ReadVariableOpReadVariableOpHidB3/kernel* 
_output_shapes
:
F*
dtype0
m

HidB3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
HidB3/bias
f
HidB3/bias/Read/ReadVariableOpReadVariableOp
HidB3/bias*
_output_shapes	
:*
dtype0
u
HidC/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:î*
shared_nameHidC/kernel
n
HidC/kernel/Read/ReadVariableOpReadVariableOpHidC/kernel*!
_output_shapes
:î*
dtype0
k
	HidC/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	HidC/bias
d
HidC/bias/Read/ReadVariableOpReadVariableOp	HidC/bias*
_output_shapes	
:*
dtype0
w
HidC2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:°*
shared_nameHidC2/kernel
p
 HidC2/kernel/Read/ReadVariableOpReadVariableOpHidC2/kernel*!
_output_shapes
:°*
dtype0
m

HidC2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
HidC2/bias
f
HidC2/bias/Read/ReadVariableOpReadVariableOp
HidC2/bias*
_output_shapes	
:*
dtype0
w
HidD1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:á*
shared_nameHidD1/kernel
p
 HidD1/kernel/Read/ReadVariableOpReadVariableOpHidD1/kernel*!
_output_shapes
:á*
dtype0
m

HidD1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
HidD1/bias
f
HidD1/bias/Read/ReadVariableOpReadVariableOp
HidD1/bias*
_output_shapes	
:*
dtype0
w
HidE1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: ð*
shared_nameHidE1/kernel
p
 HidE1/kernel/Read/ReadVariableOpReadVariableOpHidE1/kernel*!
_output_shapes
: ð*
dtype0
m

HidE1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
HidE1/bias
f
HidE1/bias/Read/ReadVariableOpReadVariableOp
HidE1/bias*
_output_shapes	
:*
dtype0
t
Hid1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameHid1/kernel
m
Hid1/kernel/Read/ReadVariableOpReadVariableOpHid1/kernel* 
_output_shapes
:
*
dtype0
k
	Hid1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	Hid1/bias
d
Hid1/bias/Read/ReadVariableOpReadVariableOp	Hid1/bias*
_output_shapes	
:*
dtype0
t
Hid2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameHid2/kernel
m
Hid2/kernel/Read/ReadVariableOpReadVariableOpHid2/kernel* 
_output_shapes
:
*
dtype0
k
	Hid2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	Hid2/bias
d
Hid2/bias/Read/ReadVariableOpReadVariableOp	Hid2/bias*
_output_shapes	
:*
dtype0
w
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(*
shared_nameoutput/kernel
p
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes
:	(*
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

Adam/ConA/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/ConA/kernel/m

&Adam/ConA/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConA/kernel/m*&
_output_shapes
:*
dtype0
x
Adam/ConA/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/ConA/bias/m
q
$Adam/ConA/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConA/bias/m*
_output_shapes
:*
dtype0

Adam/ConB/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/ConB/kernel/m

&Adam/ConB/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConB/kernel/m*&
_output_shapes
:*
dtype0
x
Adam/ConB/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/ConB/bias/m
q
$Adam/ConB/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConB/bias/m*
_output_shapes
:*
dtype0

Adam/ConC/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*#
shared_nameAdam/ConC/kernel/m

&Adam/ConC/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConC/kernel/m*&
_output_shapes
:	*
dtype0
x
Adam/ConC/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/ConC/bias/m
q
$Adam/ConC/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConC/bias/m*
_output_shapes
:*
dtype0

Adam/ConA2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConA2/kernel/m

'Adam/ConA2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConA2/kernel/m*&
_output_shapes
:*
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

Adam/ConB2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConB2/kernel/m

'Adam/ConB2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConB2/kernel/m*&
_output_shapes
:*
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

Adam/ConA3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*$
shared_nameAdam/ConA3/kernel/m

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

Adam/ConB3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*$
shared_nameAdam/ConB3/kernel/m

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

Adam/ConC2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConC2/kernel/m

'Adam/ConC2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConC2/kernel/m*&
_output_shapes
:*
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

Adam/ConD/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*#
shared_nameAdam/ConD/kernel/m

&Adam/ConD/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConD/kernel/m*&
_output_shapes
:	*
dtype0
x
Adam/ConD/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/ConD/bias/m
q
$Adam/ConD/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConD/bias/m*
_output_shapes
:*
dtype0

Adam/ConE/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/ConE/kernel/m

&Adam/ConE/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConE/kernel/m*&
_output_shapes
:*
dtype0
x
Adam/ConE/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/ConE/bias/m
q
$Adam/ConE/bias/m/Read/ReadVariableOpReadVariableOpAdam/ConE/bias/m*
_output_shapes
:*
dtype0

Adam/HidA/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ê×*#
shared_nameAdam/HidA/kernel/m
|
&Adam/HidA/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidA/kernel/m*!
_output_shapes
:Ê×*
dtype0
y
Adam/HidA/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/HidA/bias/m
r
$Adam/HidA/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidA/bias/m*
_output_shapes	
:*
dtype0

Adam/HidA2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:¤*$
shared_nameAdam/HidA2/kernel/m
~
'Adam/HidA2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidA2/kernel/m*!
_output_shapes
:¤*
dtype0
{
Adam/HidA2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/HidA2/bias/m
t
%Adam/HidA2/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidA2/bias/m*
_output_shapes	
:*
dtype0

Adam/HidA3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
F*$
shared_nameAdam/HidA3/kernel/m
}
'Adam/HidA3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidA3/kernel/m* 
_output_shapes
:
F*
dtype0
{
Adam/HidA3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/HidA3/bias/m
t
%Adam/HidA3/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidA3/bias/m*
_output_shapes	
:*
dtype0

Adam/HidB/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: ©*#
shared_nameAdam/HidB/kernel/m
|
&Adam/HidB/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidB/kernel/m*!
_output_shapes
: ©*
dtype0
y
Adam/HidB/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/HidB/bias/m
r
$Adam/HidB/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidB/bias/m*
_output_shapes	
:*
dtype0

Adam/HidB2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:¤*$
shared_nameAdam/HidB2/kernel/m
~
'Adam/HidB2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidB2/kernel/m*!
_output_shapes
:¤*
dtype0
{
Adam/HidB2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/HidB2/bias/m
t
%Adam/HidB2/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidB2/bias/m*
_output_shapes	
:*
dtype0

Adam/HidB3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
F*$
shared_nameAdam/HidB3/kernel/m
}
'Adam/HidB3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidB3/kernel/m* 
_output_shapes
:
F*
dtype0
{
Adam/HidB3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/HidB3/bias/m
t
%Adam/HidB3/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidB3/bias/m*
_output_shapes	
:*
dtype0

Adam/HidC/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:î*#
shared_nameAdam/HidC/kernel/m
|
&Adam/HidC/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidC/kernel/m*!
_output_shapes
:î*
dtype0
y
Adam/HidC/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/HidC/bias/m
r
$Adam/HidC/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidC/bias/m*
_output_shapes	
:*
dtype0

Adam/HidC2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:°*$
shared_nameAdam/HidC2/kernel/m
~
'Adam/HidC2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidC2/kernel/m*!
_output_shapes
:°*
dtype0
{
Adam/HidC2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/HidC2/bias/m
t
%Adam/HidC2/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidC2/bias/m*
_output_shapes	
:*
dtype0

Adam/HidD1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:á*$
shared_nameAdam/HidD1/kernel/m
~
'Adam/HidD1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidD1/kernel/m*!
_output_shapes
:á*
dtype0
{
Adam/HidD1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/HidD1/bias/m
t
%Adam/HidD1/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidD1/bias/m*
_output_shapes	
:*
dtype0

Adam/HidE1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: ð*$
shared_nameAdam/HidE1/kernel/m
~
'Adam/HidE1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidE1/kernel/m*!
_output_shapes
: ð*
dtype0
{
Adam/HidE1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/HidE1/bias/m
t
%Adam/HidE1/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidE1/bias/m*
_output_shapes	
:*
dtype0

Adam/Hid1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/Hid1/kernel/m
{
&Adam/Hid1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Hid1/kernel/m* 
_output_shapes
:
*
dtype0
y
Adam/Hid1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/Hid1/bias/m
r
$Adam/Hid1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Hid1/bias/m*
_output_shapes	
:*
dtype0

Adam/Hid2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/Hid2/kernel/m
{
&Adam/Hid2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Hid2/kernel/m* 
_output_shapes
:
*
dtype0
y
Adam/Hid2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/Hid2/bias/m
r
$Adam/Hid2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Hid2/bias/m*
_output_shapes	
:*
dtype0

Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(*%
shared_nameAdam/output/kernel/m
~
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*
_output_shapes
:	(*
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

Adam/ConA/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/ConA/kernel/v

&Adam/ConA/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConA/kernel/v*&
_output_shapes
:*
dtype0
x
Adam/ConA/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/ConA/bias/v
q
$Adam/ConA/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConA/bias/v*
_output_shapes
:*
dtype0

Adam/ConB/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/ConB/kernel/v

&Adam/ConB/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConB/kernel/v*&
_output_shapes
:*
dtype0
x
Adam/ConB/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/ConB/bias/v
q
$Adam/ConB/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConB/bias/v*
_output_shapes
:*
dtype0

Adam/ConC/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*#
shared_nameAdam/ConC/kernel/v

&Adam/ConC/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConC/kernel/v*&
_output_shapes
:	*
dtype0
x
Adam/ConC/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/ConC/bias/v
q
$Adam/ConC/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConC/bias/v*
_output_shapes
:*
dtype0

Adam/ConA2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConA2/kernel/v

'Adam/ConA2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConA2/kernel/v*&
_output_shapes
:*
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

Adam/ConB2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConB2/kernel/v

'Adam/ConB2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConB2/kernel/v*&
_output_shapes
:*
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

Adam/ConA3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*$
shared_nameAdam/ConA3/kernel/v

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

Adam/ConB3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*$
shared_nameAdam/ConB3/kernel/v

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

Adam/ConC2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConC2/kernel/v

'Adam/ConC2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConC2/kernel/v*&
_output_shapes
:*
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

Adam/ConD/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*#
shared_nameAdam/ConD/kernel/v

&Adam/ConD/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConD/kernel/v*&
_output_shapes
:	*
dtype0
x
Adam/ConD/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/ConD/bias/v
q
$Adam/ConD/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConD/bias/v*
_output_shapes
:*
dtype0

Adam/ConE/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/ConE/kernel/v

&Adam/ConE/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConE/kernel/v*&
_output_shapes
:*
dtype0
x
Adam/ConE/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/ConE/bias/v
q
$Adam/ConE/bias/v/Read/ReadVariableOpReadVariableOpAdam/ConE/bias/v*
_output_shapes
:*
dtype0

Adam/HidA/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ê×*#
shared_nameAdam/HidA/kernel/v
|
&Adam/HidA/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidA/kernel/v*!
_output_shapes
:Ê×*
dtype0
y
Adam/HidA/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/HidA/bias/v
r
$Adam/HidA/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidA/bias/v*
_output_shapes	
:*
dtype0

Adam/HidA2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:¤*$
shared_nameAdam/HidA2/kernel/v
~
'Adam/HidA2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidA2/kernel/v*!
_output_shapes
:¤*
dtype0
{
Adam/HidA2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/HidA2/bias/v
t
%Adam/HidA2/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidA2/bias/v*
_output_shapes	
:*
dtype0

Adam/HidA3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
F*$
shared_nameAdam/HidA3/kernel/v
}
'Adam/HidA3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidA3/kernel/v* 
_output_shapes
:
F*
dtype0
{
Adam/HidA3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/HidA3/bias/v
t
%Adam/HidA3/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidA3/bias/v*
_output_shapes	
:*
dtype0

Adam/HidB/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: ©*#
shared_nameAdam/HidB/kernel/v
|
&Adam/HidB/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidB/kernel/v*!
_output_shapes
: ©*
dtype0
y
Adam/HidB/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/HidB/bias/v
r
$Adam/HidB/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidB/bias/v*
_output_shapes	
:*
dtype0

Adam/HidB2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:¤*$
shared_nameAdam/HidB2/kernel/v
~
'Adam/HidB2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidB2/kernel/v*!
_output_shapes
:¤*
dtype0
{
Adam/HidB2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/HidB2/bias/v
t
%Adam/HidB2/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidB2/bias/v*
_output_shapes	
:*
dtype0

Adam/HidB3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
F*$
shared_nameAdam/HidB3/kernel/v
}
'Adam/HidB3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidB3/kernel/v* 
_output_shapes
:
F*
dtype0
{
Adam/HidB3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/HidB3/bias/v
t
%Adam/HidB3/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidB3/bias/v*
_output_shapes	
:*
dtype0

Adam/HidC/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:î*#
shared_nameAdam/HidC/kernel/v
|
&Adam/HidC/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidC/kernel/v*!
_output_shapes
:î*
dtype0
y
Adam/HidC/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/HidC/bias/v
r
$Adam/HidC/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidC/bias/v*
_output_shapes	
:*
dtype0

Adam/HidC2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:°*$
shared_nameAdam/HidC2/kernel/v
~
'Adam/HidC2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidC2/kernel/v*!
_output_shapes
:°*
dtype0
{
Adam/HidC2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/HidC2/bias/v
t
%Adam/HidC2/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidC2/bias/v*
_output_shapes	
:*
dtype0

Adam/HidD1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:á*$
shared_nameAdam/HidD1/kernel/v
~
'Adam/HidD1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidD1/kernel/v*!
_output_shapes
:á*
dtype0
{
Adam/HidD1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/HidD1/bias/v
t
%Adam/HidD1/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidD1/bias/v*
_output_shapes	
:*
dtype0

Adam/HidE1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: ð*$
shared_nameAdam/HidE1/kernel/v
~
'Adam/HidE1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidE1/kernel/v*!
_output_shapes
: ð*
dtype0
{
Adam/HidE1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/HidE1/bias/v
t
%Adam/HidE1/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidE1/bias/v*
_output_shapes	
:*
dtype0

Adam/Hid1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/Hid1/kernel/v
{
&Adam/Hid1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Hid1/kernel/v* 
_output_shapes
:
*
dtype0
y
Adam/Hid1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/Hid1/bias/v
r
$Adam/Hid1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Hid1/bias/v*
_output_shapes	
:*
dtype0

Adam/Hid2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/Hid2/kernel/v
{
&Adam/Hid2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Hid2/kernel/v* 
_output_shapes
:
*
dtype0
y
Adam/Hid2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/Hid2/bias/v
r
$Adam/Hid2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Hid2/bias/v*
_output_shapes	
:*
dtype0

Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(*%
shared_nameAdam/output/kernel/v
~
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*
_output_shapes
:	(*
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
Û³
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*³
value³B³ Bþ²
ñ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-5
layer-14
layer_with_weights-6
layer-15
layer_with_weights-7
layer-16
layer_with_weights-8
layer-17
layer_with_weights-9
layer-18
layer-19
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
*layer_with_weights-10
*layer-41
+layer_with_weights-11
+layer-42
,layer_with_weights-12
,layer-43
-layer_with_weights-13
-layer-44
.layer_with_weights-14
.layer-45
/layer_with_weights-15
/layer-46
0layer_with_weights-16
0layer-47
1layer_with_weights-17
1layer-48
2layer_with_weights-18
2layer-49
3layer_with_weights-19
3layer-50
4layer-51
5layer_with_weights-20
5layer-52
6layer_with_weights-21
6layer-53
7layer_with_weights-22
7layer-54
8	optimizer
9
signatures
#:_self_saveable_object_factories
;regularization_losses
<	variables
=trainable_variables
>	keras_api
%
#?_self_saveable_object_factories


@kernel
Abias
#B_self_saveable_object_factories
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api


Gkernel
Hbias
#I_self_saveable_object_factories
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
w
#N_self_saveable_object_factories
Oregularization_losses
P	variables
Qtrainable_variables
R	keras_api
w
#S_self_saveable_object_factories
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
w
#X_self_saveable_object_factories
Yregularization_losses
Z	variables
[trainable_variables
\	keras_api
w
#]_self_saveable_object_factories
^regularization_losses
_	variables
`trainable_variables
a	keras_api


bkernel
cbias
#d_self_saveable_object_factories
eregularization_losses
f	variables
gtrainable_variables
h	keras_api


ikernel
jbias
#k_self_saveable_object_factories
lregularization_losses
m	variables
ntrainable_variables
o	keras_api


pkernel
qbias
#r_self_saveable_object_factories
sregularization_losses
t	variables
utrainable_variables
v	keras_api
w
#w_self_saveable_object_factories
xregularization_losses
y	variables
ztrainable_variables
{	keras_api
x
#|_self_saveable_object_factories
}regularization_losses
~	variables
trainable_variables
	keras_api
|
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
|
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api

kernel
	bias
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api

kernel
	bias
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api

kernel
	bias
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api

 kernel
	¡bias
$¢_self_saveable_object_factories
£regularization_losses
¤	variables
¥trainable_variables
¦	keras_api

§kernel
	¨bias
$©_self_saveable_object_factories
ªregularization_losses
«	variables
¬trainable_variables
­	keras_api
|
$®_self_saveable_object_factories
¯regularization_losses
°	variables
±trainable_variables
²	keras_api
|
$³_self_saveable_object_factories
´regularization_losses
µ	variables
¶trainable_variables
·	keras_api
|
$¸_self_saveable_object_factories
¹regularization_losses
º	variables
»trainable_variables
¼	keras_api
|
$½_self_saveable_object_factories
¾regularization_losses
¿	variables
Àtrainable_variables
Á	keras_api
|
$Â_self_saveable_object_factories
Ãregularization_losses
Ä	variables
Åtrainable_variables
Æ	keras_api
|
$Ç_self_saveable_object_factories
Èregularization_losses
É	variables
Êtrainable_variables
Ë	keras_api
|
$Ì_self_saveable_object_factories
Íregularization_losses
Î	variables
Ïtrainable_variables
Ð	keras_api
|
$Ñ_self_saveable_object_factories
Òregularization_losses
Ó	variables
Ôtrainable_variables
Õ	keras_api
|
$Ö_self_saveable_object_factories
×regularization_losses
Ø	variables
Ùtrainable_variables
Ú	keras_api
|
$Û_self_saveable_object_factories
Üregularization_losses
Ý	variables
Þtrainable_variables
ß	keras_api
|
$à_self_saveable_object_factories
áregularization_losses
â	variables
ãtrainable_variables
ä	keras_api
|
$å_self_saveable_object_factories
æregularization_losses
ç	variables
ètrainable_variables
é	keras_api
|
$ê_self_saveable_object_factories
ëregularization_losses
ì	variables
ítrainable_variables
î	keras_api
|
$ï_self_saveable_object_factories
ðregularization_losses
ñ	variables
òtrainable_variables
ó	keras_api
|
$ô_self_saveable_object_factories
õregularization_losses
ö	variables
÷trainable_variables
ø	keras_api
|
$ù_self_saveable_object_factories
úregularization_losses
û	variables
ütrainable_variables
ý	keras_api
|
$þ_self_saveable_object_factories
ÿregularization_losses
	variables
trainable_variables
	keras_api
|
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
|
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
|
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
|
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
|
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api

kernel
	bias
$_self_saveable_object_factories
regularization_losses
 	variables
¡trainable_variables
¢	keras_api

£kernel
	¤bias
$¥_self_saveable_object_factories
¦regularization_losses
§	variables
¨trainable_variables
©	keras_api

ªkernel
	«bias
$¬_self_saveable_object_factories
­regularization_losses
®	variables
¯trainable_variables
°	keras_api

±kernel
	²bias
$³_self_saveable_object_factories
´regularization_losses
µ	variables
¶trainable_variables
·	keras_api

¸kernel
	¹bias
$º_self_saveable_object_factories
»regularization_losses
¼	variables
½trainable_variables
¾	keras_api

¿kernel
	Àbias
$Á_self_saveable_object_factories
Âregularization_losses
Ã	variables
Ätrainable_variables
Å	keras_api

Ækernel
	Çbias
$È_self_saveable_object_factories
Éregularization_losses
Ê	variables
Ëtrainable_variables
Ì	keras_api

Íkernel
	Îbias
$Ï_self_saveable_object_factories
Ðregularization_losses
Ñ	variables
Òtrainable_variables
Ó	keras_api

Ôkernel
	Õbias
$Ö_self_saveable_object_factories
×regularization_losses
Ø	variables
Ùtrainable_variables
Ú	keras_api

Ûkernel
	Übias
$Ý_self_saveable_object_factories
Þregularization_losses
ß	variables
àtrainable_variables
á	keras_api
|
$â_self_saveable_object_factories
ãregularization_losses
ä	variables
åtrainable_variables
æ	keras_api

çkernel
	èbias
$é_self_saveable_object_factories
êregularization_losses
ë	variables
ìtrainable_variables
í	keras_api

îkernel
	ïbias
$ð_self_saveable_object_factories
ñregularization_losses
ò	variables
ótrainable_variables
ô	keras_api

õkernel
	öbias
$÷_self_saveable_object_factories
øregularization_losses
ù	variables
útrainable_variables
û	keras_api
¥
	üiter
ýbeta_1
þbeta_2

ÿdecay
learning_rate@mAm Gm¡Hm¢bm£cm¤im¥jm¦pm§qm¨	m©	mª	m«	m¬	m­	m®	 m¯	¡m°	§m±	¨m²	m³	m´	£mµ	¤m¶	ªm·	«m¸	±m¹	²mº	¸m»	¹m¼	¿m½	Àm¾	Æm¿	ÇmÀ	ÍmÁ	ÎmÂ	ÔmÃ	ÕmÄ	ÛmÅ	ÜmÆ	çmÇ	èmÈ	îmÉ	ïmÊ	õmË	ömÌ@vÍAvÎGvÏHvÐbvÑcvÒivÓjvÔpvÕqvÖ	v×	vØ	vÙ	vÚ	vÛ	vÜ	 vÝ	¡vÞ	§vß	¨và	vá	vâ	£vã	¤vä	ªvå	«væ	±vç	²vè	¸vé	¹vê	¿vë	Àvì	Æví	Çvî	Ívï	Îvð	Ôvñ	Õvò	Ûvó	Üvô	çvõ	èvö	îv÷	ïvø	õvù	övú
 
 
 

@0
A1
G2
H3
b4
c5
i6
j7
p8
q9
10
11
12
13
14
15
 16
¡17
§18
¨19
20
21
£22
¤23
ª24
«25
±26
²27
¸28
¹29
¿30
À31
Æ32
Ç33
Í34
Î35
Ô36
Õ37
Û38
Ü39
ç40
è41
î42
ï43
õ44
ö45

@0
A1
G2
H3
b4
c5
i6
j7
p8
q9
10
11
12
13
14
15
 16
¡17
§18
¨19
20
21
£22
¤23
ª24
«25
±26
²27
¸28
¹29
¿30
À31
Æ32
Ç33
Í34
Î35
Ô36
Õ37
Û38
Ü39
ç40
è41
î42
ï43
õ44
ö45
²
;regularization_losses
layers
 layer_regularization_losses
non_trainable_variables
<	variables
layer_metrics
metrics
=trainable_variables
 
WU
VARIABLE_VALUEConA/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	ConA/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

@0
A1

@0
A1
²
Cregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
D	variables
layer_metrics
metrics
Etrainable_variables
WU
VARIABLE_VALUEConB/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	ConB/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

G0
H1

G0
H1
²
Jregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
K	variables
layer_metrics
metrics
Ltrainable_variables
 
 
 
 
²
Oregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
P	variables
layer_metrics
metrics
Qtrainable_variables
 
 
 
 
²
Tregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
U	variables
layer_metrics
metrics
Vtrainable_variables
 
 
 
 
²
Yregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
Z	variables
layer_metrics
metrics
[trainable_variables
 
 
 
 
²
^regularization_losses
layers
  layer_regularization_losses
¡non_trainable_variables
_	variables
¢layer_metrics
£metrics
`trainable_variables
WU
VARIABLE_VALUEConC/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	ConC/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

b0
c1

b0
c1
²
eregularization_losses
¤layers
 ¥layer_regularization_losses
¦non_trainable_variables
f	variables
§layer_metrics
¨metrics
gtrainable_variables
XV
VARIABLE_VALUEConA2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
ConA2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

i0
j1

i0
j1
²
lregularization_losses
©layers
 ªlayer_regularization_losses
«non_trainable_variables
m	variables
¬layer_metrics
­metrics
ntrainable_variables
XV
VARIABLE_VALUEConB2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
ConB2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

p0
q1

p0
q1
²
sregularization_losses
®layers
 ¯layer_regularization_losses
°non_trainable_variables
t	variables
±layer_metrics
²metrics
utrainable_variables
 
 
 
 
²
xregularization_losses
³layers
 ´layer_regularization_losses
µnon_trainable_variables
y	variables
¶layer_metrics
·metrics
ztrainable_variables
 
 
 
 
²
}regularization_losses
¸layers
 ¹layer_regularization_losses
ºnon_trainable_variables
~	variables
»layer_metrics
¼metrics
trainable_variables
 
 
 
 
µ
regularization_losses
½layers
 ¾layer_regularization_losses
¿non_trainable_variables
	variables
Àlayer_metrics
Ámetrics
trainable_variables
 
 
 
 
µ
regularization_losses
Âlayers
 Ãlayer_regularization_losses
Änon_trainable_variables
	variables
Ålayer_metrics
Æmetrics
trainable_variables
XV
VARIABLE_VALUEConA3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
ConA3/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
µ
regularization_losses
Çlayers
 Èlayer_regularization_losses
Énon_trainable_variables
	variables
Êlayer_metrics
Ëmetrics
trainable_variables
XV
VARIABLE_VALUEConB3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
ConB3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
µ
regularization_losses
Ìlayers
 Ílayer_regularization_losses
Înon_trainable_variables
	variables
Ïlayer_metrics
Ðmetrics
trainable_variables
XV
VARIABLE_VALUEConC2/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
ConC2/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
µ
regularization_losses
Ñlayers
 Òlayer_regularization_losses
Ónon_trainable_variables
	variables
Ôlayer_metrics
Õmetrics
trainable_variables
WU
VARIABLE_VALUEConD/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	ConD/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

 0
¡1

 0
¡1
µ
£regularization_losses
Ölayers
 ×layer_regularization_losses
Ønon_trainable_variables
¤	variables
Ùlayer_metrics
Úmetrics
¥trainable_variables
WU
VARIABLE_VALUEConE/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	ConE/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

§0
¨1

§0
¨1
µ
ªregularization_losses
Ûlayers
 Ülayer_regularization_losses
Ýnon_trainable_variables
«	variables
Þlayer_metrics
ßmetrics
¬trainable_variables
 
 
 
 
µ
¯regularization_losses
àlayers
 álayer_regularization_losses
ânon_trainable_variables
°	variables
ãlayer_metrics
ämetrics
±trainable_variables
 
 
 
 
µ
´regularization_losses
ålayers
 ælayer_regularization_losses
çnon_trainable_variables
µ	variables
èlayer_metrics
émetrics
¶trainable_variables
 
 
 
 
µ
¹regularization_losses
êlayers
 ëlayer_regularization_losses
ìnon_trainable_variables
º	variables
ílayer_metrics
îmetrics
»trainable_variables
 
 
 
 
µ
¾regularization_losses
ïlayers
 ðlayer_regularization_losses
ñnon_trainable_variables
¿	variables
òlayer_metrics
ómetrics
Àtrainable_variables
 
 
 
 
µ
Ãregularization_losses
ôlayers
 õlayer_regularization_losses
önon_trainable_variables
Ä	variables
÷layer_metrics
ømetrics
Åtrainable_variables
 
 
 
 
µ
Èregularization_losses
ùlayers
 úlayer_regularization_losses
ûnon_trainable_variables
É	variables
ülayer_metrics
ýmetrics
Êtrainable_variables
 
 
 
 
µ
Íregularization_losses
þlayers
 ÿlayer_regularization_losses
non_trainable_variables
Î	variables
layer_metrics
metrics
Ïtrainable_variables
 
 
 
 
µ
Òregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
Ó	variables
layer_metrics
metrics
Ôtrainable_variables
 
 
 
 
µ
×regularization_losses
layers
 layer_regularization_losses
non_trainable_variables
Ø	variables
layer_metrics
metrics
Ùtrainable_variables
 
 
 
 
µ
Üregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
Ý	variables
layer_metrics
metrics
Þtrainable_variables
 
 
 
 
µ
áregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
â	variables
layer_metrics
metrics
ãtrainable_variables
 
 
 
 
µ
æregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
ç	variables
layer_metrics
metrics
ètrainable_variables
 
 
 
 
µ
ëregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
ì	variables
layer_metrics
 metrics
ítrainable_variables
 
 
 
 
µ
ðregularization_losses
¡layers
 ¢layer_regularization_losses
£non_trainable_variables
ñ	variables
¤layer_metrics
¥metrics
òtrainable_variables
 
 
 
 
µ
õregularization_losses
¦layers
 §layer_regularization_losses
¨non_trainable_variables
ö	variables
©layer_metrics
ªmetrics
÷trainable_variables
 
 
 
 
µ
úregularization_losses
«layers
 ¬layer_regularization_losses
­non_trainable_variables
û	variables
®layer_metrics
¯metrics
ütrainable_variables
 
 
 
 
µ
ÿregularization_losses
°layers
 ±layer_regularization_losses
²non_trainable_variables
	variables
³layer_metrics
´metrics
trainable_variables
 
 
 
 
µ
regularization_losses
µlayers
 ¶layer_regularization_losses
·non_trainable_variables
	variables
¸layer_metrics
¹metrics
trainable_variables
 
 
 
 
µ
regularization_losses
ºlayers
 »layer_regularization_losses
¼non_trainable_variables
	variables
½layer_metrics
¾metrics
trainable_variables
 
 
 
 
µ
regularization_losses
¿layers
 Àlayer_regularization_losses
Ánon_trainable_variables
	variables
Âlayer_metrics
Ãmetrics
trainable_variables
 
 
 
 
µ
regularization_losses
Älayers
 Ålayer_regularization_losses
Ænon_trainable_variables
	variables
Çlayer_metrics
Èmetrics
trainable_variables
 
 
 
 
µ
regularization_losses
Élayers
 Êlayer_regularization_losses
Ënon_trainable_variables
	variables
Ìlayer_metrics
Ímetrics
trainable_variables
XV
VARIABLE_VALUEHidA/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	HidA/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
µ
regularization_losses
Îlayers
 Ïlayer_regularization_losses
Ðnon_trainable_variables
 	variables
Ñlayer_metrics
Òmetrics
¡trainable_variables
YW
VARIABLE_VALUEHidA2/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidA2/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

£0
¤1

£0
¤1
µ
¦regularization_losses
Ólayers
 Ôlayer_regularization_losses
Õnon_trainable_variables
§	variables
Ölayer_metrics
×metrics
¨trainable_variables
YW
VARIABLE_VALUEHidA3/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidA3/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

ª0
«1

ª0
«1
µ
­regularization_losses
Ølayers
 Ùlayer_regularization_losses
Únon_trainable_variables
®	variables
Ûlayer_metrics
Ümetrics
¯trainable_variables
XV
VARIABLE_VALUEHidB/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	HidB/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

±0
²1

±0
²1
µ
´regularization_losses
Ýlayers
 Þlayer_regularization_losses
ßnon_trainable_variables
µ	variables
àlayer_metrics
ámetrics
¶trainable_variables
YW
VARIABLE_VALUEHidB2/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidB2/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

¸0
¹1

¸0
¹1
µ
»regularization_losses
âlayers
 ãlayer_regularization_losses
änon_trainable_variables
¼	variables
ålayer_metrics
æmetrics
½trainable_variables
YW
VARIABLE_VALUEHidB3/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidB3/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

¿0
À1

¿0
À1
µ
Âregularization_losses
çlayers
 èlayer_regularization_losses
énon_trainable_variables
Ã	variables
êlayer_metrics
ëmetrics
Ätrainable_variables
XV
VARIABLE_VALUEHidC/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	HidC/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

Æ0
Ç1

Æ0
Ç1
µ
Éregularization_losses
ìlayers
 ílayer_regularization_losses
înon_trainable_variables
Ê	variables
ïlayer_metrics
ðmetrics
Ëtrainable_variables
YW
VARIABLE_VALUEHidC2/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidC2/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

Í0
Î1

Í0
Î1
µ
Ðregularization_losses
ñlayers
 òlayer_regularization_losses
ónon_trainable_variables
Ñ	variables
ôlayer_metrics
õmetrics
Òtrainable_variables
YW
VARIABLE_VALUEHidD1/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidD1/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

Ô0
Õ1

Ô0
Õ1
µ
×regularization_losses
ölayers
 ÷layer_regularization_losses
ønon_trainable_variables
Ø	variables
ùlayer_metrics
úmetrics
Ùtrainable_variables
YW
VARIABLE_VALUEHidE1/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidE1/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

Û0
Ü1

Û0
Ü1
µ
Þregularization_losses
ûlayers
 ülayer_regularization_losses
ýnon_trainable_variables
ß	variables
þlayer_metrics
ÿmetrics
àtrainable_variables
 
 
 
 
µ
ãregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
ä	variables
layer_metrics
metrics
åtrainable_variables
XV
VARIABLE_VALUEHid1/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	Hid1/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

ç0
è1

ç0
è1
µ
êregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
ë	variables
layer_metrics
metrics
ìtrainable_variables
XV
VARIABLE_VALUEHid2/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	Hid2/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

î0
ï1

î0
ï1
µ
ñregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
ò	variables
layer_metrics
metrics
ótrainable_variables
ZX
VARIABLE_VALUEoutput/kernel7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEoutput/bias5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

õ0
ö1

õ0
ö1
µ
øregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
ù	variables
layer_metrics
metrics
útrainable_variables
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
®
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
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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

total

count
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables
zx
VARIABLE_VALUEAdam/ConA/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConA/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConB/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConB/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConC/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConC/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConA2/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConA2/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConB2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConB2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConA3/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConA3/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConB3/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConB3/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConC2/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConC2/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConD/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConD/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConE/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConE/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/HidA/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/HidA/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidA2/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidA2/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidA3/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidA3/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/HidB/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/HidB/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidB2/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidB2/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidB3/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidB3/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/HidC/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/HidC/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidC2/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidC2/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidD1/kernel/mSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidD1/bias/mQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidE1/kernel/mSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidE1/bias/mQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/Hid1/kernel/mSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Hid1/bias/mQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/Hid2/kernel/mSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Hid2/bias/mQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/output/kernel/mSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/output/bias/mQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConA/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConA/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConB/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConB/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConC/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConC/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConA2/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConA2/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConB2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConB2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConA3/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConA3/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConB3/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConB3/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConC2/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConC2/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConD/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConD/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConE/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConE/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/HidA/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/HidA/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidA2/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidA2/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidA3/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidA3/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/HidB/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/HidB/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidB2/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidB2/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidB3/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidB3/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/HidC/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/HidC/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidC2/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidC2/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidD1/kernel/vSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidD1/bias/vQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidE1/kernel/vSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidE1/bias/vQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/Hid1/kernel/vSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Hid1/bias/vQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/Hid2/kernel/vSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Hid2/bias/vQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/output/kernel/vSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/output/bias/vQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_inputPlaceholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿßÁ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿßÁ
Ó
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputConB/kernel	ConB/biasConA/kernel	ConA/biasConC/kernel	ConC/biasConB2/kernel
ConB2/biasConA2/kernel
ConA2/biasConE/kernel	ConE/biasConD/kernel	ConD/biasConC2/kernel
ConC2/biasConB3/kernel
ConB3/biasConA3/kernel
ConA3/biasHidA/kernel	HidA/biasHidA2/kernel
HidA2/biasHidA3/kernel
HidA3/biasHidB/kernel	HidB/biasHidB2/kernel
HidB2/biasHidB3/kernel
HidB3/biasHidC/kernel	HidC/biasHidC2/kernel
HidC2/biasHidD1/kernel
HidD1/biasHidE1/kernel
HidE1/biasHid1/kernel	Hid1/biasHid2/kernel	Hid2/biasoutput/kerneloutput/bias*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_183041
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
æ-
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameConA/kernel/Read/ReadVariableOpConA/bias/Read/ReadVariableOpConB/kernel/Read/ReadVariableOpConB/bias/Read/ReadVariableOpConC/kernel/Read/ReadVariableOpConC/bias/Read/ReadVariableOp ConA2/kernel/Read/ReadVariableOpConA2/bias/Read/ReadVariableOp ConB2/kernel/Read/ReadVariableOpConB2/bias/Read/ReadVariableOp ConA3/kernel/Read/ReadVariableOpConA3/bias/Read/ReadVariableOp ConB3/kernel/Read/ReadVariableOpConB3/bias/Read/ReadVariableOp ConC2/kernel/Read/ReadVariableOpConC2/bias/Read/ReadVariableOpConD/kernel/Read/ReadVariableOpConD/bias/Read/ReadVariableOpConE/kernel/Read/ReadVariableOpConE/bias/Read/ReadVariableOpHidA/kernel/Read/ReadVariableOpHidA/bias/Read/ReadVariableOp HidA2/kernel/Read/ReadVariableOpHidA2/bias/Read/ReadVariableOp HidA3/kernel/Read/ReadVariableOpHidA3/bias/Read/ReadVariableOpHidB/kernel/Read/ReadVariableOpHidB/bias/Read/ReadVariableOp HidB2/kernel/Read/ReadVariableOpHidB2/bias/Read/ReadVariableOp HidB3/kernel/Read/ReadVariableOpHidB3/bias/Read/ReadVariableOpHidC/kernel/Read/ReadVariableOpHidC/bias/Read/ReadVariableOp HidC2/kernel/Read/ReadVariableOpHidC2/bias/Read/ReadVariableOp HidD1/kernel/Read/ReadVariableOpHidD1/bias/Read/ReadVariableOp HidE1/kernel/Read/ReadVariableOpHidE1/bias/Read/ReadVariableOpHid1/kernel/Read/ReadVariableOpHid1/bias/Read/ReadVariableOpHid2/kernel/Read/ReadVariableOpHid2/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp&Adam/ConA/kernel/m/Read/ReadVariableOp$Adam/ConA/bias/m/Read/ReadVariableOp&Adam/ConB/kernel/m/Read/ReadVariableOp$Adam/ConB/bias/m/Read/ReadVariableOp&Adam/ConC/kernel/m/Read/ReadVariableOp$Adam/ConC/bias/m/Read/ReadVariableOp'Adam/ConA2/kernel/m/Read/ReadVariableOp%Adam/ConA2/bias/m/Read/ReadVariableOp'Adam/ConB2/kernel/m/Read/ReadVariableOp%Adam/ConB2/bias/m/Read/ReadVariableOp'Adam/ConA3/kernel/m/Read/ReadVariableOp%Adam/ConA3/bias/m/Read/ReadVariableOp'Adam/ConB3/kernel/m/Read/ReadVariableOp%Adam/ConB3/bias/m/Read/ReadVariableOp'Adam/ConC2/kernel/m/Read/ReadVariableOp%Adam/ConC2/bias/m/Read/ReadVariableOp&Adam/ConD/kernel/m/Read/ReadVariableOp$Adam/ConD/bias/m/Read/ReadVariableOp&Adam/ConE/kernel/m/Read/ReadVariableOp$Adam/ConE/bias/m/Read/ReadVariableOp&Adam/HidA/kernel/m/Read/ReadVariableOp$Adam/HidA/bias/m/Read/ReadVariableOp'Adam/HidA2/kernel/m/Read/ReadVariableOp%Adam/HidA2/bias/m/Read/ReadVariableOp'Adam/HidA3/kernel/m/Read/ReadVariableOp%Adam/HidA3/bias/m/Read/ReadVariableOp&Adam/HidB/kernel/m/Read/ReadVariableOp$Adam/HidB/bias/m/Read/ReadVariableOp'Adam/HidB2/kernel/m/Read/ReadVariableOp%Adam/HidB2/bias/m/Read/ReadVariableOp'Adam/HidB3/kernel/m/Read/ReadVariableOp%Adam/HidB3/bias/m/Read/ReadVariableOp&Adam/HidC/kernel/m/Read/ReadVariableOp$Adam/HidC/bias/m/Read/ReadVariableOp'Adam/HidC2/kernel/m/Read/ReadVariableOp%Adam/HidC2/bias/m/Read/ReadVariableOp'Adam/HidD1/kernel/m/Read/ReadVariableOp%Adam/HidD1/bias/m/Read/ReadVariableOp'Adam/HidE1/kernel/m/Read/ReadVariableOp%Adam/HidE1/bias/m/Read/ReadVariableOp&Adam/Hid1/kernel/m/Read/ReadVariableOp$Adam/Hid1/bias/m/Read/ReadVariableOp&Adam/Hid2/kernel/m/Read/ReadVariableOp$Adam/Hid2/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp&Adam/ConA/kernel/v/Read/ReadVariableOp$Adam/ConA/bias/v/Read/ReadVariableOp&Adam/ConB/kernel/v/Read/ReadVariableOp$Adam/ConB/bias/v/Read/ReadVariableOp&Adam/ConC/kernel/v/Read/ReadVariableOp$Adam/ConC/bias/v/Read/ReadVariableOp'Adam/ConA2/kernel/v/Read/ReadVariableOp%Adam/ConA2/bias/v/Read/ReadVariableOp'Adam/ConB2/kernel/v/Read/ReadVariableOp%Adam/ConB2/bias/v/Read/ReadVariableOp'Adam/ConA3/kernel/v/Read/ReadVariableOp%Adam/ConA3/bias/v/Read/ReadVariableOp'Adam/ConB3/kernel/v/Read/ReadVariableOp%Adam/ConB3/bias/v/Read/ReadVariableOp'Adam/ConC2/kernel/v/Read/ReadVariableOp%Adam/ConC2/bias/v/Read/ReadVariableOp&Adam/ConD/kernel/v/Read/ReadVariableOp$Adam/ConD/bias/v/Read/ReadVariableOp&Adam/ConE/kernel/v/Read/ReadVariableOp$Adam/ConE/bias/v/Read/ReadVariableOp&Adam/HidA/kernel/v/Read/ReadVariableOp$Adam/HidA/bias/v/Read/ReadVariableOp'Adam/HidA2/kernel/v/Read/ReadVariableOp%Adam/HidA2/bias/v/Read/ReadVariableOp'Adam/HidA3/kernel/v/Read/ReadVariableOp%Adam/HidA3/bias/v/Read/ReadVariableOp&Adam/HidB/kernel/v/Read/ReadVariableOp$Adam/HidB/bias/v/Read/ReadVariableOp'Adam/HidB2/kernel/v/Read/ReadVariableOp%Adam/HidB2/bias/v/Read/ReadVariableOp'Adam/HidB3/kernel/v/Read/ReadVariableOp%Adam/HidB3/bias/v/Read/ReadVariableOp&Adam/HidC/kernel/v/Read/ReadVariableOp$Adam/HidC/bias/v/Read/ReadVariableOp'Adam/HidC2/kernel/v/Read/ReadVariableOp%Adam/HidC2/bias/v/Read/ReadVariableOp'Adam/HidD1/kernel/v/Read/ReadVariableOp%Adam/HidD1/bias/v/Read/ReadVariableOp'Adam/HidE1/kernel/v/Read/ReadVariableOp%Adam/HidE1/bias/v/Read/ReadVariableOp&Adam/Hid1/kernel/v/Read/ReadVariableOp$Adam/Hid1/bias/v/Read/ReadVariableOp&Adam/Hid2/kernel/v/Read/ReadVariableOp$Adam/Hid2/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*£
Tin
2	*
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
GPU2*0J 8 *(
f#R!
__inference__traced_save_185252
å
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConA/kernel	ConA/biasConB/kernel	ConB/biasConC/kernel	ConC/biasConA2/kernel
ConA2/biasConB2/kernel
ConB2/biasConA3/kernel
ConA3/biasConB3/kernel
ConB3/biasConC2/kernel
ConC2/biasConD/kernel	ConD/biasConE/kernel	ConE/biasHidA/kernel	HidA/biasHidA2/kernel
HidA2/biasHidA3/kernel
HidA3/biasHidB/kernel	HidB/biasHidB2/kernel
HidB2/biasHidB3/kernel
HidB3/biasHidC/kernel	HidC/biasHidC2/kernel
HidC2/biasHidD1/kernel
HidD1/biasHidE1/kernel
HidE1/biasHid1/kernel	Hid1/biasHid2/kernel	Hid2/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/ConA/kernel/mAdam/ConA/bias/mAdam/ConB/kernel/mAdam/ConB/bias/mAdam/ConC/kernel/mAdam/ConC/bias/mAdam/ConA2/kernel/mAdam/ConA2/bias/mAdam/ConB2/kernel/mAdam/ConB2/bias/mAdam/ConA3/kernel/mAdam/ConA3/bias/mAdam/ConB3/kernel/mAdam/ConB3/bias/mAdam/ConC2/kernel/mAdam/ConC2/bias/mAdam/ConD/kernel/mAdam/ConD/bias/mAdam/ConE/kernel/mAdam/ConE/bias/mAdam/HidA/kernel/mAdam/HidA/bias/mAdam/HidA2/kernel/mAdam/HidA2/bias/mAdam/HidA3/kernel/mAdam/HidA3/bias/mAdam/HidB/kernel/mAdam/HidB/bias/mAdam/HidB2/kernel/mAdam/HidB2/bias/mAdam/HidB3/kernel/mAdam/HidB3/bias/mAdam/HidC/kernel/mAdam/HidC/bias/mAdam/HidC2/kernel/mAdam/HidC2/bias/mAdam/HidD1/kernel/mAdam/HidD1/bias/mAdam/HidE1/kernel/mAdam/HidE1/bias/mAdam/Hid1/kernel/mAdam/Hid1/bias/mAdam/Hid2/kernel/mAdam/Hid2/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/ConA/kernel/vAdam/ConA/bias/vAdam/ConB/kernel/vAdam/ConB/bias/vAdam/ConC/kernel/vAdam/ConC/bias/vAdam/ConA2/kernel/vAdam/ConA2/bias/vAdam/ConB2/kernel/vAdam/ConB2/bias/vAdam/ConA3/kernel/vAdam/ConA3/bias/vAdam/ConB3/kernel/vAdam/ConB3/bias/vAdam/ConC2/kernel/vAdam/ConC2/bias/vAdam/ConD/kernel/vAdam/ConD/bias/vAdam/ConE/kernel/vAdam/ConE/bias/vAdam/HidA/kernel/vAdam/HidA/bias/vAdam/HidA2/kernel/vAdam/HidA2/bias/vAdam/HidA3/kernel/vAdam/HidA3/bias/vAdam/HidB/kernel/vAdam/HidB/bias/vAdam/HidB2/kernel/vAdam/HidB2/bias/vAdam/HidB3/kernel/vAdam/HidB3/bias/vAdam/HidC/kernel/vAdam/HidC/bias/vAdam/HidC2/kernel/vAdam/HidC2/bias/vAdam/HidD1/kernel/vAdam/HidD1/bias/vAdam/HidE1/kernel/vAdam/HidE1/bias/vAdam/Hid1/kernel/vAdam/Hid1/bias/vAdam/Hid2/kernel/vAdam/Hid2/bias/vAdam/output/kernel/vAdam/output/bias/v*¢
Tin
2*
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
GPU2*0J 8 *+
f&R$
"__inference__traced_restore_185703Ùì"

c
E__inference_dropout_7_layer_call_and_return_conditional_losses_181180

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
è
ú
A__inference_ConA3_layer_call_and_return_conditional_losses_181111

inputs8
conv2d_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
¸
]
A__inference_PoolA_layer_call_and_return_conditional_losses_183769

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿
 
_user_specified_nameinputs
Ñ

A__inference_model_layer_call_and_return_conditional_losses_181515

inputs%
conb_180908:
conb_180910:%
cona_180925:
cona_180927:%
conc_180954:	
conc_180956:&
conb2_180991:
conb2_180993:&
cona2_181008:
cona2_181010:%
cone_181044:
cone_181046:%
cond_181061:	
cond_181063:&
conc2_181078:
conc2_181080:&
conb3_181095:(
conb3_181097:(&
cona3_181112:(
cona3_181114:( 
hida_181288:Ê×
hida_181290:	!
hida2_181305:¤
hida2_181307:	 
hida3_181322:
F
hida3_181324:	 
hidb_181339: ©
hidb_181341:	!
hidb2_181356:¤
hidb2_181358:	 
hidb3_181373:
F
hidb3_181375:	 
hidc_181390:î
hidc_181392:	!
hidc2_181407:°
hidc2_181409:	!
hidd1_181424:á
hidd1_181426:	!
hide1_181441: ð
hide1_181443:	
hid1_181475:

hid1_181477:	
hid2_181492:

hid2_181494:	 
output_181509:	(
output_181511:(
identity¢ConA/StatefulPartitionedCall¢ConA2/StatefulPartitionedCall¢ConA3/StatefulPartitionedCall¢ConB/StatefulPartitionedCall¢ConB2/StatefulPartitionedCall¢ConB3/StatefulPartitionedCall¢ConC/StatefulPartitionedCall¢ConC2/StatefulPartitionedCall¢ConD/StatefulPartitionedCall¢ConE/StatefulPartitionedCall¢Hid1/StatefulPartitionedCall¢Hid2/StatefulPartitionedCall¢HidA/StatefulPartitionedCall¢HidA2/StatefulPartitionedCall¢HidA3/StatefulPartitionedCall¢HidB/StatefulPartitionedCall¢HidB2/StatefulPartitionedCall¢HidB3/StatefulPartitionedCall¢HidC/StatefulPartitionedCall¢HidC2/StatefulPartitionedCall¢HidD1/StatefulPartitionedCall¢HidE1/StatefulPartitionedCall¢output/StatefulPartitionedCall
ConB/StatefulPartitionedCallStatefulPartitionedCallinputsconb_180908conb_180910*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ConB_layer_call_and_return_conditional_losses_1809072
ConB/StatefulPartitionedCall
ConA/StatefulPartitionedCallStatefulPartitionedCallinputscona_180925cona_180927*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ConA_layer_call_and_return_conditional_losses_1809242
ConA/StatefulPartitionedCallõ
PoolB/PartitionedCallPartitionedCall%ConB/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolB_layer_call_and_return_conditional_losses_1809342
PoolB/PartitionedCallõ
PoolA/PartitionedCallPartitionedCall%ConA/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolA_layer_call_and_return_conditional_losses_1809402
PoolA/PartitionedCall
ConC/StatefulPartitionedCallStatefulPartitionedCallinputsconc_180954conc_180956*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ConC_layer_call_and_return_conditional_losses_1809532
ConC/StatefulPartitionedCallú
dropout_1/PartitionedCallPartitionedCallPoolB/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1809642
dropout_1/PartitionedCallô
dropout/PartitionedCallPartitionedCallPoolA/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1809712
dropout/PartitionedCallõ
PoolC/PartitionedCallPartitionedCall%ConC/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolC_layer_call_and_return_conditional_losses_1809772
PoolC/PartitionedCall¬
ConB2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conb2_180991conb2_180993*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_ConB2_layer_call_and_return_conditional_losses_1809902
ConB2/StatefulPartitionedCallª
ConA2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0cona2_181008cona2_181010*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_ConA2_layer_call_and_return_conditional_losses_1810072
ConA2/StatefulPartitionedCallú
dropout_2/PartitionedCallPartitionedCallPoolC/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_1810182
dropout_2/PartitionedCallù
PoolB2/PartitionedCallPartitionedCall&ConB2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolB2_layer_call_and_return_conditional_losses_1810242
PoolB2/PartitionedCallù
PoolA2/PartitionedCallPartitionedCall&ConA2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolA2_layer_call_and_return_conditional_losses_1810302
PoolA2/PartitionedCall
ConE/StatefulPartitionedCallStatefulPartitionedCallinputscone_181044cone_181046*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ConE_layer_call_and_return_conditional_losses_1810432
ConE/StatefulPartitionedCall
ConD/StatefulPartitionedCallStatefulPartitionedCallinputscond_181061cond_181063*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ConD_layer_call_and_return_conditional_losses_1810602
ConD/StatefulPartitionedCall¬
ConC2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0conc2_181078conc2_181080*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_ConC2_layer_call_and_return_conditional_losses_1810772
ConC2/StatefulPartitionedCall©
ConB3/StatefulPartitionedCallStatefulPartitionedCallPoolB2/PartitionedCall:output:0conb3_181095conb3_181097*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_ConB3_layer_call_and_return_conditional_losses_1810942
ConB3/StatefulPartitionedCall©
ConA3/StatefulPartitionedCallStatefulPartitionedCallPoolA2/PartitionedCall:output:0cona3_181112cona3_181114*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_ConA3_layer_call_and_return_conditional_losses_1811112
ConA3/StatefulPartitionedCallõ
PoolE/PartitionedCallPartitionedCall%ConE/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolE_layer_call_and_return_conditional_losses_1811212
PoolE/PartitionedCallõ
PoolD/PartitionedCallPartitionedCall%ConD/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolD_layer_call_and_return_conditional_losses_1811272
PoolD/PartitionedCallù
PoolC2/PartitionedCallPartitionedCall&ConC2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolC2_layer_call_and_return_conditional_losses_1811332
PoolC2/PartitionedCallù
PoolB3/PartitionedCallPartitionedCall&ConB3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolB3_layer_call_and_return_conditional_losses_1811392
PoolB3/PartitionedCallù
PoolA3/PartitionedCallPartitionedCall&ConA3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolA3_layer_call_and_return_conditional_losses_1811452
PoolA3/PartitionedCallú
dropout_4/PartitionedCallPartitionedCallPoolE/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1811522
dropout_4/PartitionedCallú
dropout_3/PartitionedCallPartitionedCallPoolD/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_1811592
dropout_3/PartitionedCallû
dropout_9/PartitionedCallPartitionedCallPoolC2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_1811662
dropout_9/PartitionedCallû
dropout_8/PartitionedCallPartitionedCallPoolB3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_1811732
dropout_8/PartitionedCallû
dropout_7/PartitionedCallPartitionedCallPoolB2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_1811802
dropout_7/PartitionedCallû
dropout_6/PartitionedCallPartitionedCallPoolA3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_1811872
dropout_6/PartitionedCallû
dropout_5/PartitionedCallPartitionedCallPoolA2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_1811942
dropout_5/PartitionedCallø
flatten_9/PartitionedCallPartitionedCall"dropout_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ð* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_9_layer_call_and_return_conditional_losses_1812022
flatten_9/PartitionedCallø
flatten_8/PartitionedCallPartitionedCall"dropout_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿá* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_1812102
flatten_8/PartitionedCallø
flatten_7/PartitionedCallPartitionedCall"dropout_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_1812182
flatten_7/PartitionedCallø
flatten_6/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿî* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_1812262
flatten_6/PartitionedCall÷
flatten_5/PartitionedCallPartitionedCall"dropout_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_1812342
flatten_5/PartitionedCallø
flatten_4/PartitionedCallPartitionedCall"dropout_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_1812422
flatten_4/PartitionedCallô
flatten_3/PartitionedCallPartitionedCallPoolB/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1812502
flatten_3/PartitionedCall÷
flatten_2/PartitionedCallPartitionedCall"dropout_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_1812582
flatten_2/PartitionedCallø
flatten_1/PartitionedCallPartitionedCall"dropout_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1812662
flatten_1/PartitionedCallî
flatten/PartitionedCallPartitionedCallPoolA/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ×* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1812742
flatten/PartitionedCall
HidA/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0hida_181288hida_181290*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_HidA_layer_call_and_return_conditional_losses_1812872
HidA/StatefulPartitionedCall¥
HidA2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0hida2_181305hida2_181307*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidA2_layer_call_and_return_conditional_losses_1813042
HidA2/StatefulPartitionedCall¥
HidA3/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0hida3_181322hida3_181324*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidA3_layer_call_and_return_conditional_losses_1813212
HidA3/StatefulPartitionedCall 
HidB/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0hidb_181339hidb_181341*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_HidB_layer_call_and_return_conditional_losses_1813382
HidB/StatefulPartitionedCall¥
HidB2/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0hidb2_181356hidb2_181358*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidB2_layer_call_and_return_conditional_losses_1813552
HidB2/StatefulPartitionedCall¥
HidB3/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0hidb3_181373hidb3_181375*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidB3_layer_call_and_return_conditional_losses_1813722
HidB3/StatefulPartitionedCall 
HidC/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0hidc_181390hidc_181392*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_HidC_layer_call_and_return_conditional_losses_1813892
HidC/StatefulPartitionedCall¥
HidC2/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0hidc2_181407hidc2_181409*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidC2_layer_call_and_return_conditional_losses_1814062
HidC2/StatefulPartitionedCall¥
HidD1/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0hidd1_181424hidd1_181426*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidD1_layer_call_and_return_conditional_losses_1814232
HidD1/StatefulPartitionedCall¥
HidE1/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0hide1_181441hide1_181443*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidE1_layer_call_and_return_conditional_losses_1814402
HidE1/StatefulPartitionedCallï
concatenate/PartitionedCallPartitionedCall%HidA/StatefulPartitionedCall:output:0&HidA2/StatefulPartitionedCall:output:0&HidA3/StatefulPartitionedCall:output:0%HidB/StatefulPartitionedCall:output:0&HidB2/StatefulPartitionedCall:output:0&HidB3/StatefulPartitionedCall:output:0%HidC/StatefulPartitionedCall:output:0&HidC2/StatefulPartitionedCall:output:0&HidD1/StatefulPartitionedCall:output:0&HidE1/StatefulPartitionedCall:output:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_1814612
concatenate/PartitionedCall¢
Hid1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0hid1_181475hid1_181477*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Hid1_layer_call_and_return_conditional_losses_1814742
Hid1/StatefulPartitionedCall£
Hid2/StatefulPartitionedCallStatefulPartitionedCall%Hid1/StatefulPartitionedCall:output:0hid2_181492hid2_181494*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Hid2_layer_call_and_return_conditional_losses_1814912
Hid2/StatefulPartitionedCall¬
output/StatefulPartitionedCallStatefulPartitionedCall%Hid2/StatefulPartitionedCall:output:0output_181509output_181511*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1815082 
output/StatefulPartitionedCall
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity¥
NoOpNoOp^ConA/StatefulPartitionedCall^ConA2/StatefulPartitionedCall^ConA3/StatefulPartitionedCall^ConB/StatefulPartitionedCall^ConB2/StatefulPartitionedCall^ConB3/StatefulPartitionedCall^ConC/StatefulPartitionedCall^ConC2/StatefulPartitionedCall^ConD/StatefulPartitionedCall^ConE/StatefulPartitionedCall^Hid1/StatefulPartitionedCall^Hid2/StatefulPartitionedCall^HidA/StatefulPartitionedCall^HidA2/StatefulPartitionedCall^HidA3/StatefulPartitionedCall^HidB/StatefulPartitionedCall^HidB2/StatefulPartitionedCall^HidB3/StatefulPartitionedCall^HidC/StatefulPartitionedCall^HidC2/StatefulPartitionedCall^HidD1/StatefulPartitionedCall^HidE1/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes{
y:ÿÿÿÿÿÿÿÿÿßÁ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
ConA/StatefulPartitionedCallConA/StatefulPartitionedCall2>
ConA2/StatefulPartitionedCallConA2/StatefulPartitionedCall2>
ConA3/StatefulPartitionedCallConA3/StatefulPartitionedCall2<
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
HidA3/StatefulPartitionedCallHidA3/StatefulPartitionedCall2<
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
:ÿÿÿÿÿÿÿÿÿßÁ
 
_user_specified_nameinputs
á
F
*__inference_dropout_1_layer_call_fn_183848

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1809642
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿH?:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?
 
_user_specified_nameinputs
¸
]
A__inference_PoolE_layer_call_and_return_conditional_losses_184190

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÍ³:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³
 
_user_specified_nameinputs


&__inference_ConB2_layer_call_fn_183913

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_ConB2_layer_call_and_return_conditional_losses_1809902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿH?: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?
 
_user_specified_nameinputs

ö
A__inference_HidE1_layer_call_and_return_conditional_losses_181440

inputs3
matmul_readvariableop_resource: ð.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
: ð*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ð: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ð
 
_user_specified_nameinputs

c
E__inference_dropout_4_layer_call_and_return_conditional_losses_181152

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿD;:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;
 
_user_specified_nameinputs
¡
]
A__inference_PoolA_layer_call_and_return_conditional_losses_180678

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
]
A__inference_PoolD_layer_call_and_return_conditional_losses_181127

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹
 
_user_specified_nameinputs
¡
]
A__inference_PoolA_layer_call_and_return_conditional_losses_183764

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ
B
&__inference_PoolC_layer_call_fn_183928

inputs
identityå
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolC_layer_call_and_return_conditional_losses_1807222
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
d
E__inference_dropout_6_layer_call_and_return_conditional_losses_184244

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs

ö
A__inference_HidB2_layer_call_and_return_conditional_losses_184590

inputs3
matmul_readvariableop_resource:¤.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:¤*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¤: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
 
_user_specified_nameinputs

c
E__inference_dropout_8_layer_call_and_return_conditional_losses_184286

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
¢
^
B__inference_PoolA3_layer_call_and_return_conditional_losses_184105

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
^
B__inference_PoolC2_layer_call_and_return_conditional_losses_184150

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿE<:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<
 
_user_specified_nameinputs


&__inference_ConB3_layer_call_fn_184040

inputs!
unknown:(
	unknown_0:(
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_ConB3_layer_call_and_return_conditional_losses_1810942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ#: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
¸
]
A__inference_PoolB_layer_call_and_return_conditional_losses_180934

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÙ½:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½
 
_user_specified_nameinputs
Þß
ÿ
A__inference_model_layer_call_and_return_conditional_losses_183525

inputs=
#conb_conv2d_readvariableop_resource:2
$conb_biasadd_readvariableop_resource:=
#cona_conv2d_readvariableop_resource:2
$cona_biasadd_readvariableop_resource:=
#conc_conv2d_readvariableop_resource:	2
$conc_biasadd_readvariableop_resource:>
$conb2_conv2d_readvariableop_resource:3
%conb2_biasadd_readvariableop_resource:>
$cona2_conv2d_readvariableop_resource:3
%cona2_biasadd_readvariableop_resource:=
#cone_conv2d_readvariableop_resource:2
$cone_biasadd_readvariableop_resource:=
#cond_conv2d_readvariableop_resource:	2
$cond_biasadd_readvariableop_resource:>
$conc2_conv2d_readvariableop_resource:3
%conc2_biasadd_readvariableop_resource:>
$conb3_conv2d_readvariableop_resource:(3
%conb3_biasadd_readvariableop_resource:(>
$cona3_conv2d_readvariableop_resource:(3
%cona3_biasadd_readvariableop_resource:(8
#hida_matmul_readvariableop_resource:Ê×3
$hida_biasadd_readvariableop_resource:	9
$hida2_matmul_readvariableop_resource:¤4
%hida2_biasadd_readvariableop_resource:	8
$hida3_matmul_readvariableop_resource:
F4
%hida3_biasadd_readvariableop_resource:	8
#hidb_matmul_readvariableop_resource: ©3
$hidb_biasadd_readvariableop_resource:	9
$hidb2_matmul_readvariableop_resource:¤4
%hidb2_biasadd_readvariableop_resource:	8
$hidb3_matmul_readvariableop_resource:
F4
%hidb3_biasadd_readvariableop_resource:	8
#hidc_matmul_readvariableop_resource:î3
$hidc_biasadd_readvariableop_resource:	9
$hidc2_matmul_readvariableop_resource:°4
%hidc2_biasadd_readvariableop_resource:	9
$hidd1_matmul_readvariableop_resource:á4
%hidd1_biasadd_readvariableop_resource:	9
$hide1_matmul_readvariableop_resource: ð4
%hide1_biasadd_readvariableop_resource:	7
#hid1_matmul_readvariableop_resource:
3
$hid1_biasadd_readvariableop_resource:	7
#hid2_matmul_readvariableop_resource:
3
$hid2_biasadd_readvariableop_resource:	8
%output_matmul_readvariableop_resource:	(4
&output_biasadd_readvariableop_resource:(
identity¢ConA/BiasAdd/ReadVariableOp¢ConA/Conv2D/ReadVariableOp¢ConA2/BiasAdd/ReadVariableOp¢ConA2/Conv2D/ReadVariableOp¢ConA3/BiasAdd/ReadVariableOp¢ConA3/Conv2D/ReadVariableOp¢ConB/BiasAdd/ReadVariableOp¢ConB/Conv2D/ReadVariableOp¢ConB2/BiasAdd/ReadVariableOp¢ConB2/Conv2D/ReadVariableOp¢ConB3/BiasAdd/ReadVariableOp¢ConB3/Conv2D/ReadVariableOp¢ConC/BiasAdd/ReadVariableOp¢ConC/Conv2D/ReadVariableOp¢ConC2/BiasAdd/ReadVariableOp¢ConC2/Conv2D/ReadVariableOp¢ConD/BiasAdd/ReadVariableOp¢ConD/Conv2D/ReadVariableOp¢ConE/BiasAdd/ReadVariableOp¢ConE/Conv2D/ReadVariableOp¢Hid1/BiasAdd/ReadVariableOp¢Hid1/MatMul/ReadVariableOp¢Hid2/BiasAdd/ReadVariableOp¢Hid2/MatMul/ReadVariableOp¢HidA/BiasAdd/ReadVariableOp¢HidA/MatMul/ReadVariableOp¢HidA2/BiasAdd/ReadVariableOp¢HidA2/MatMul/ReadVariableOp¢HidA3/BiasAdd/ReadVariableOp¢HidA3/MatMul/ReadVariableOp¢HidB/BiasAdd/ReadVariableOp¢HidB/MatMul/ReadVariableOp¢HidB2/BiasAdd/ReadVariableOp¢HidB2/MatMul/ReadVariableOp¢HidB3/BiasAdd/ReadVariableOp¢HidB3/MatMul/ReadVariableOp¢HidC/BiasAdd/ReadVariableOp¢HidC/MatMul/ReadVariableOp¢HidC2/BiasAdd/ReadVariableOp¢HidC2/MatMul/ReadVariableOp¢HidD1/BiasAdd/ReadVariableOp¢HidD1/MatMul/ReadVariableOp¢HidE1/BiasAdd/ReadVariableOp¢HidE1/MatMul/ReadVariableOp¢output/BiasAdd/ReadVariableOp¢output/MatMul/ReadVariableOp¤
ConB/Conv2D/ReadVariableOpReadVariableOp#conb_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConB/Conv2D/ReadVariableOpµ
ConB/Conv2DConv2Dinputs"ConB/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½*
paddingVALID*
strides
2
ConB/Conv2D
ConB/BiasAdd/ReadVariableOpReadVariableOp$conb_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConB/BiasAdd/ReadVariableOp
ConB/BiasAddBiasAddConB/Conv2D:output:0#ConB/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½2
ConB/BiasAddq
	ConB/ReluReluConB/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½2
	ConB/Relu¤
ConA/Conv2D/ReadVariableOpReadVariableOp#cona_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConA/Conv2D/ReadVariableOpµ
ConA/Conv2DConv2Dinputs"ConA/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿*
paddingVALID*
strides
2
ConA/Conv2D
ConA/BiasAdd/ReadVariableOpReadVariableOp$cona_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConA/BiasAdd/ReadVariableOp
ConA/BiasAddBiasAddConA/Conv2D:output:0#ConA/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿2
ConA/BiasAddq
	ConA/ReluReluConA/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿2
	ConA/Relu¯
PoolB/MaxPoolMaxPoolConB/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?*
ksize
*
paddingVALID*
strides
2
PoolB/MaxPool¯
PoolA/MaxPoolMaxPoolConA/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?*
ksize
*
paddingVALID*
strides
2
PoolA/MaxPool¤
ConC/Conv2D/ReadVariableOpReadVariableOp#conc_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
ConC/Conv2D/ReadVariableOpµ
ConC/Conv2DConv2Dinputs"ConC/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»*
paddingVALID*
strides
2
ConC/Conv2D
ConC/BiasAdd/ReadVariableOpReadVariableOp$conc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConC/BiasAdd/ReadVariableOp
ConC/BiasAddBiasAddConC/Conv2D:output:0#ConC/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»2
ConC/BiasAddq
	ConC/ReluReluConC/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»2
	ConC/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_1/dropout/Const©
dropout_1/dropout/MulMulPoolB/MaxPool:output:0 dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?2
dropout_1/dropout/Mulx
dropout_1/dropout/ShapeShapePoolB/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/ShapeÚ
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2"
 dropout_1/dropout/GreaterEqual/yî
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?2 
dropout_1/dropout/GreaterEqual¥
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?2
dropout_1/dropout/Castª
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?2
dropout_1/dropout/Mul_1s
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/dropout/Const£
dropout/dropout/MulMulPoolA/MaxPool:output:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?2
dropout/dropout/Mult
dropout/dropout/ShapeShapePoolA/MaxPool:output:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeÔ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2 
dropout/dropout/GreaterEqual/yæ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?2
dropout/dropout/Cast¢
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?2
dropout/dropout/Mul_1¯
PoolC/MaxPoolMaxPoolConC/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>*
ksize
*
paddingVALID*
strides
2
PoolC/MaxPool§
ConB2/Conv2D/ReadVariableOpReadVariableOp$conb2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConB2/Conv2D/ReadVariableOpË
ConB2/Conv2DConv2Ddropout_1/dropout/Mul_1:z:0#ConB2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=*
paddingVALID*
strides
2
ConB2/Conv2D
ConB2/BiasAdd/ReadVariableOpReadVariableOp%conb2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConB2/BiasAdd/ReadVariableOp 
ConB2/BiasAddBiasAddConB2/Conv2D:output:0$ConB2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=2
ConB2/BiasAddr

ConB2/ReluReluConB2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=2

ConB2/Relu§
ConA2/Conv2D/ReadVariableOpReadVariableOp$cona2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConA2/Conv2D/ReadVariableOpÉ
ConA2/Conv2DConv2Ddropout/dropout/Mul_1:z:0#ConA2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=*
paddingVALID*
strides
2
ConA2/Conv2D
ConA2/BiasAdd/ReadVariableOpReadVariableOp%cona2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConA2/BiasAdd/ReadVariableOp 
ConA2/BiasAddBiasAddConA2/Conv2D:output:0$ConA2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2
ConA2/BiasAddr

ConA2/ReluReluConA2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2

ConA2/Reluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_2/dropout/Const©
dropout_2/dropout/MulMulPoolC/MaxPool:output:0 dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>2
dropout_2/dropout/Mulx
dropout_2/dropout/ShapeShapePoolC/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_2/dropout/ShapeÚ
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2"
 dropout_2/dropout/GreaterEqual/yî
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>2 
dropout_2/dropout/GreaterEqual¥
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>2
dropout_2/dropout/Castª
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>2
dropout_2/dropout/Mul_1²
PoolB2/MaxPoolMaxPoolConB2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
ksize
*
paddingVALID*
strides
2
PoolB2/MaxPool²
PoolA2/MaxPoolMaxPoolConA2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
ksize
*
paddingVALID*
strides
2
PoolA2/MaxPool¤
ConE/Conv2D/ReadVariableOpReadVariableOp#cone_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConE/Conv2D/ReadVariableOpµ
ConE/Conv2DConv2Dinputs"ConE/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³*
paddingVALID*
strides
2
ConE/Conv2D
ConE/BiasAdd/ReadVariableOpReadVariableOp$cone_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConE/BiasAdd/ReadVariableOp
ConE/BiasAddBiasAddConE/Conv2D:output:0#ConE/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³2
ConE/BiasAddq
	ConE/ReluReluConE/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³2
	ConE/Relu¤
ConD/Conv2D/ReadVariableOpReadVariableOp#cond_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
ConD/Conv2D/ReadVariableOpµ
ConD/Conv2DConv2Dinputs"ConD/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹*
paddingVALID*
strides
2
ConD/Conv2D
ConD/BiasAdd/ReadVariableOpReadVariableOp$cond_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConD/BiasAdd/ReadVariableOp
ConD/BiasAddBiasAddConD/Conv2D:output:0#ConD/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹2
ConD/BiasAddq
	ConD/ReluReluConD/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹2
	ConD/Relu§
ConC2/Conv2D/ReadVariableOpReadVariableOp$conc2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConC2/Conv2D/ReadVariableOpË
ConC2/Conv2DConv2Ddropout_2/dropout/Mul_1:z:0#ConC2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<*
paddingVALID*
strides
2
ConC2/Conv2D
ConC2/BiasAdd/ReadVariableOpReadVariableOp%conc2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConC2/BiasAdd/ReadVariableOp 
ConC2/BiasAddBiasAddConC2/Conv2D:output:0$ConC2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<2
ConC2/BiasAddr

ConC2/ReluReluConC2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<2

ConC2/Relu§
ConB3/Conv2D/ReadVariableOpReadVariableOp$conb3_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
ConB3/Conv2D/ReadVariableOpÇ
ConB3/Conv2DConv2DPoolB2/MaxPool:output:0#ConB3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(*
paddingVALID*
strides
2
ConB3/Conv2D
ConB3/BiasAdd/ReadVariableOpReadVariableOp%conb3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
ConB3/BiasAdd/ReadVariableOp 
ConB3/BiasAddBiasAddConB3/Conv2D:output:0$ConB3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2
ConB3/BiasAddr

ConB3/ReluReluConB3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2

ConB3/Relu§
ConA3/Conv2D/ReadVariableOpReadVariableOp$cona3_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
ConA3/Conv2D/ReadVariableOpÇ
ConA3/Conv2DConv2DPoolA2/MaxPool:output:0#ConA3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(*
paddingVALID*
strides
2
ConA3/Conv2D
ConA3/BiasAdd/ReadVariableOpReadVariableOp%cona3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
ConA3/BiasAdd/ReadVariableOp 
ConA3/BiasAddBiasAddConA3/Conv2D:output:0$ConA3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2
ConA3/BiasAddr

ConA3/ReluReluConA3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2

ConA3/Relu¯
PoolE/MaxPoolMaxPoolConE/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;*
ksize
*
paddingVALID*
strides
2
PoolE/MaxPool¯
PoolD/MaxPoolMaxPoolConD/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=*
ksize
*
paddingVALID*
strides
2
PoolD/MaxPool²
PoolC2/MaxPoolMaxPoolConC2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"*
ksize
*
paddingVALID*
strides
2
PoolC2/MaxPool²
PoolB3/MaxPoolMaxPoolConB3/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
ksize
*
paddingVALID*
strides
2
PoolB3/MaxPool²
PoolA3/MaxPoolMaxPoolConA3/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
ksize
*
paddingVALID*
strides
2
PoolA3/MaxPoolw
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_4/dropout/Const©
dropout_4/dropout/MulMulPoolE/MaxPool:output:0 dropout_4/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;2
dropout_4/dropout/Mulx
dropout_4/dropout/ShapeShapePoolE/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_4/dropout/ShapeÚ
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;*
dtype020
.dropout_4/dropout/random_uniform/RandomUniform
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2"
 dropout_4/dropout/GreaterEqual/yî
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;2 
dropout_4/dropout/GreaterEqual¥
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;2
dropout_4/dropout/Castª
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;2
dropout_4/dropout/Mul_1w
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_3/dropout/Const©
dropout_3/dropout/MulMulPoolD/MaxPool:output:0 dropout_3/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2
dropout_3/dropout/Mulx
dropout_3/dropout/ShapeShapePoolD/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_3/dropout/ShapeÚ
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2"
 dropout_3/dropout/GreaterEqual/yî
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2 
dropout_3/dropout/GreaterEqual¥
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2
dropout_3/dropout/Castª
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2
dropout_3/dropout/Mul_1w
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_9/dropout/Constª
dropout_9/dropout/MulMulPoolC2/MaxPool:output:0 dropout_9/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2
dropout_9/dropout/Muly
dropout_9/dropout/ShapeShapePoolC2/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_9/dropout/ShapeÚ
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"*
dtype020
.dropout_9/dropout/random_uniform/RandomUniform
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2"
 dropout_9/dropout/GreaterEqual/yî
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2 
dropout_9/dropout/GreaterEqual¥
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2
dropout_9/dropout/Castª
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2
dropout_9/dropout/Mul_1w
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_8/dropout/Constª
dropout_8/dropout/MulMulPoolB3/MaxPool:output:0 dropout_8/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dropout_8/dropout/Muly
dropout_8/dropout/ShapeShapePoolB3/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_8/dropout/ShapeÚ
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
dtype020
.dropout_8/dropout/random_uniform/RandomUniform
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2"
 dropout_8/dropout/GreaterEqual/yî
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2 
dropout_8/dropout/GreaterEqual¥
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dropout_8/dropout/Castª
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dropout_8/dropout/Mul_1w
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_7/dropout/Constª
dropout_7/dropout/MulMulPoolB2/MaxPool:output:0 dropout_7/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
dropout_7/dropout/Muly
dropout_7/dropout/ShapeShapePoolB2/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_7/dropout/ShapeÚ
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
dtype020
.dropout_7/dropout/random_uniform/RandomUniform
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2"
 dropout_7/dropout/GreaterEqual/yî
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2 
dropout_7/dropout/GreaterEqual¥
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
dropout_7/dropout/Castª
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
dropout_7/dropout/Mul_1w
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_6/dropout/Constª
dropout_6/dropout/MulMulPoolA3/MaxPool:output:0 dropout_6/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dropout_6/dropout/Muly
dropout_6/dropout/ShapeShapePoolA3/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_6/dropout/ShapeÚ
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
dtype020
.dropout_6/dropout/random_uniform/RandomUniform
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2"
 dropout_6/dropout/GreaterEqual/yî
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2 
dropout_6/dropout/GreaterEqual¥
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dropout_6/dropout/Castª
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dropout_6/dropout/Mul_1w
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_5/dropout/Constª
dropout_5/dropout/MulMulPoolA2/MaxPool:output:0 dropout_5/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
dropout_5/dropout/Muly
dropout_5/dropout/ShapeShapePoolA2/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_5/dropout/ShapeÚ
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
dtype020
.dropout_5/dropout/random_uniform/RandomUniform
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2"
 dropout_5/dropout/GreaterEqual/yî
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2 
dropout_5/dropout/GreaterEqual¥
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
dropout_5/dropout/Castª
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
dropout_5/dropout/Mul_1s
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ x 2
flatten_9/Const
flatten_9/ReshapeReshapedropout_4/dropout/Mul_1:z:0flatten_9/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ð2
flatten_9/Reshapes
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ0 2
flatten_8/Const
flatten_8/ReshapeReshapedropout_3/dropout/Mul_1:z:0flatten_8/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿá2
flatten_8/Reshapes
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ°O  2
flatten_7/Const
flatten_7/ReshapeReshapedropout_9/dropout/Mul_1:z:0flatten_7/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
flatten_7/Reshapes
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿî 2
flatten_6/Const
flatten_6/ReshapeReshapedropout_2/dropout/Mul_1:z:0flatten_6/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿî2
flatten_6/Reshapes
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ #  2
flatten_5/Const
flatten_5/ReshapeReshapedropout_8/dropout/Mul_1:z:0flatten_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
flatten_5/Reshapes
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿR  2
flatten_4/Const
flatten_4/ReshapeReshapedropout_7/dropout/Mul_1:z:0flatten_4/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤2
flatten_4/Reshapes
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ Ô  2
flatten_3/Const
flatten_3/ReshapeReshapePoolB/MaxPool:output:0flatten_3/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©2
flatten_3/Reshapes
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ #  2
flatten_2/Const
flatten_2/ReshapeReshapedropout_6/dropout/Mul_1:z:0flatten_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
flatten_2/Reshapes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿR  2
flatten_1/Const
flatten_1/ReshapeReshapedropout_5/dropout/Mul_1:z:0flatten_1/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤2
flatten_1/Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÊk  2
flatten/Const
flatten/ReshapeReshapePoolA/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ×2
flatten/Reshape
HidA/MatMul/ReadVariableOpReadVariableOp#hida_matmul_readvariableop_resource*!
_output_shapes
:Ê×*
dtype02
HidA/MatMul/ReadVariableOp
HidA/MatMulMatMulflatten/Reshape:output:0"HidA/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidA/MatMul
HidA/BiasAdd/ReadVariableOpReadVariableOp$hida_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
HidA/BiasAdd/ReadVariableOp
HidA/BiasAddBiasAddHidA/MatMul:product:0#HidA/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidA/BiasAddh
	HidA/ReluReluHidA/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	HidA/Relu¢
HidA2/MatMul/ReadVariableOpReadVariableOp$hida2_matmul_readvariableop_resource*!
_output_shapes
:¤*
dtype02
HidA2/MatMul/ReadVariableOp
HidA2/MatMulMatMulflatten_1/Reshape:output:0#HidA2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidA2/MatMul
HidA2/BiasAdd/ReadVariableOpReadVariableOp%hida2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
HidA2/BiasAdd/ReadVariableOp
HidA2/BiasAddBiasAddHidA2/MatMul:product:0$HidA2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidA2/BiasAddk

HidA2/ReluReluHidA2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

HidA2/Relu¡
HidA3/MatMul/ReadVariableOpReadVariableOp$hida3_matmul_readvariableop_resource* 
_output_shapes
:
F*
dtype02
HidA3/MatMul/ReadVariableOp
HidA3/MatMulMatMulflatten_2/Reshape:output:0#HidA3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidA3/MatMul
HidA3/BiasAdd/ReadVariableOpReadVariableOp%hida3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
HidA3/BiasAdd/ReadVariableOp
HidA3/BiasAddBiasAddHidA3/MatMul:product:0$HidA3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidA3/BiasAddk

HidA3/ReluReluHidA3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

HidA3/Relu
HidB/MatMul/ReadVariableOpReadVariableOp#hidb_matmul_readvariableop_resource*!
_output_shapes
: ©*
dtype02
HidB/MatMul/ReadVariableOp
HidB/MatMulMatMulflatten_3/Reshape:output:0"HidB/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidB/MatMul
HidB/BiasAdd/ReadVariableOpReadVariableOp$hidb_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
HidB/BiasAdd/ReadVariableOp
HidB/BiasAddBiasAddHidB/MatMul:product:0#HidB/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidB/BiasAddh
	HidB/ReluReluHidB/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	HidB/Relu¢
HidB2/MatMul/ReadVariableOpReadVariableOp$hidb2_matmul_readvariableop_resource*!
_output_shapes
:¤*
dtype02
HidB2/MatMul/ReadVariableOp
HidB2/MatMulMatMulflatten_4/Reshape:output:0#HidB2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidB2/MatMul
HidB2/BiasAdd/ReadVariableOpReadVariableOp%hidb2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
HidB2/BiasAdd/ReadVariableOp
HidB2/BiasAddBiasAddHidB2/MatMul:product:0$HidB2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidB2/BiasAddk

HidB2/ReluReluHidB2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

HidB2/Relu¡
HidB3/MatMul/ReadVariableOpReadVariableOp$hidb3_matmul_readvariableop_resource* 
_output_shapes
:
F*
dtype02
HidB3/MatMul/ReadVariableOp
HidB3/MatMulMatMulflatten_5/Reshape:output:0#HidB3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidB3/MatMul
HidB3/BiasAdd/ReadVariableOpReadVariableOp%hidb3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
HidB3/BiasAdd/ReadVariableOp
HidB3/BiasAddBiasAddHidB3/MatMul:product:0$HidB3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidB3/BiasAddk

HidB3/ReluReluHidB3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

HidB3/Relu
HidC/MatMul/ReadVariableOpReadVariableOp#hidc_matmul_readvariableop_resource*!
_output_shapes
:î*
dtype02
HidC/MatMul/ReadVariableOp
HidC/MatMulMatMulflatten_6/Reshape:output:0"HidC/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidC/MatMul
HidC/BiasAdd/ReadVariableOpReadVariableOp$hidc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
HidC/BiasAdd/ReadVariableOp
HidC/BiasAddBiasAddHidC/MatMul:product:0#HidC/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidC/BiasAddh
	HidC/ReluReluHidC/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	HidC/Relu¢
HidC2/MatMul/ReadVariableOpReadVariableOp$hidc2_matmul_readvariableop_resource*!
_output_shapes
:°*
dtype02
HidC2/MatMul/ReadVariableOp
HidC2/MatMulMatMulflatten_7/Reshape:output:0#HidC2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidC2/MatMul
HidC2/BiasAdd/ReadVariableOpReadVariableOp%hidc2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
HidC2/BiasAdd/ReadVariableOp
HidC2/BiasAddBiasAddHidC2/MatMul:product:0$HidC2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidC2/BiasAddk

HidC2/ReluReluHidC2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

HidC2/Relu¢
HidD1/MatMul/ReadVariableOpReadVariableOp$hidd1_matmul_readvariableop_resource*!
_output_shapes
:á*
dtype02
HidD1/MatMul/ReadVariableOp
HidD1/MatMulMatMulflatten_8/Reshape:output:0#HidD1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidD1/MatMul
HidD1/BiasAdd/ReadVariableOpReadVariableOp%hidd1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
HidD1/BiasAdd/ReadVariableOp
HidD1/BiasAddBiasAddHidD1/MatMul:product:0$HidD1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidD1/BiasAddk

HidD1/ReluReluHidD1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

HidD1/Relu¢
HidE1/MatMul/ReadVariableOpReadVariableOp$hide1_matmul_readvariableop_resource*!
_output_shapes
: ð*
dtype02
HidE1/MatMul/ReadVariableOp
HidE1/MatMulMatMulflatten_9/Reshape:output:0#HidE1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidE1/MatMul
HidE1/BiasAdd/ReadVariableOpReadVariableOp%hide1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
HidE1/BiasAdd/ReadVariableOp
HidE1/BiasAddBiasAddHidE1/MatMul:product:0$HidE1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidE1/BiasAddk

HidE1/ReluReluHidE1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

HidE1/Relut
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis
concatenate/concatConcatV2HidA/Relu:activations:0HidA2/Relu:activations:0HidA3/Relu:activations:0HidB/Relu:activations:0HidB2/Relu:activations:0HidB3/Relu:activations:0HidC/Relu:activations:0HidC2/Relu:activations:0HidD1/Relu:activations:0HidE1/Relu:activations:0 concatenate/concat/axis:output:0*
N
*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatenate/concat
Hid1/MatMul/ReadVariableOpReadVariableOp#hid1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
Hid1/MatMul/ReadVariableOp
Hid1/MatMulMatMulconcatenate/concat:output:0"Hid1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Hid1/MatMul
Hid1/BiasAdd/ReadVariableOpReadVariableOp$hid1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
Hid1/BiasAdd/ReadVariableOp
Hid1/BiasAddBiasAddHid1/MatMul:product:0#Hid1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Hid1/BiasAddh
	Hid1/ReluReluHid1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Hid1/Relu
Hid2/MatMul/ReadVariableOpReadVariableOp#hid2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
Hid2/MatMul/ReadVariableOp
Hid2/MatMulMatMulHid1/Relu:activations:0"Hid2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Hid2/MatMul
Hid2/BiasAdd/ReadVariableOpReadVariableOp$hid2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
Hid2/BiasAdd/ReadVariableOp
Hid2/BiasAddBiasAddHid2/MatMul:product:0#Hid2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Hid2/BiasAddh
	Hid2/TanhTanhHid2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Hid2/Tanh£
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	(*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMulHid2/Tanh:y:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
output/MatMul¡
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
output/Softmaxs
IdentityIdentityoutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity·
NoOpNoOp^ConA/BiasAdd/ReadVariableOp^ConA/Conv2D/ReadVariableOp^ConA2/BiasAdd/ReadVariableOp^ConA2/Conv2D/ReadVariableOp^ConA3/BiasAdd/ReadVariableOp^ConA3/Conv2D/ReadVariableOp^ConB/BiasAdd/ReadVariableOp^ConB/Conv2D/ReadVariableOp^ConB2/BiasAdd/ReadVariableOp^ConB2/Conv2D/ReadVariableOp^ConB3/BiasAdd/ReadVariableOp^ConB3/Conv2D/ReadVariableOp^ConC/BiasAdd/ReadVariableOp^ConC/Conv2D/ReadVariableOp^ConC2/BiasAdd/ReadVariableOp^ConC2/Conv2D/ReadVariableOp^ConD/BiasAdd/ReadVariableOp^ConD/Conv2D/ReadVariableOp^ConE/BiasAdd/ReadVariableOp^ConE/Conv2D/ReadVariableOp^Hid1/BiasAdd/ReadVariableOp^Hid1/MatMul/ReadVariableOp^Hid2/BiasAdd/ReadVariableOp^Hid2/MatMul/ReadVariableOp^HidA/BiasAdd/ReadVariableOp^HidA/MatMul/ReadVariableOp^HidA2/BiasAdd/ReadVariableOp^HidA2/MatMul/ReadVariableOp^HidA3/BiasAdd/ReadVariableOp^HidA3/MatMul/ReadVariableOp^HidB/BiasAdd/ReadVariableOp^HidB/MatMul/ReadVariableOp^HidB2/BiasAdd/ReadVariableOp^HidB2/MatMul/ReadVariableOp^HidB3/BiasAdd/ReadVariableOp^HidB3/MatMul/ReadVariableOp^HidC/BiasAdd/ReadVariableOp^HidC/MatMul/ReadVariableOp^HidC2/BiasAdd/ReadVariableOp^HidC2/MatMul/ReadVariableOp^HidD1/BiasAdd/ReadVariableOp^HidD1/MatMul/ReadVariableOp^HidE1/BiasAdd/ReadVariableOp^HidE1/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes{
y:ÿÿÿÿÿÿÿÿÿßÁ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
ConA/BiasAdd/ReadVariableOpConA/BiasAdd/ReadVariableOp28
ConA/Conv2D/ReadVariableOpConA/Conv2D/ReadVariableOp2<
ConA2/BiasAdd/ReadVariableOpConA2/BiasAdd/ReadVariableOp2:
ConA2/Conv2D/ReadVariableOpConA2/Conv2D/ReadVariableOp2<
ConA3/BiasAdd/ReadVariableOpConA3/BiasAdd/ReadVariableOp2:
ConA3/Conv2D/ReadVariableOpConA3/Conv2D/ReadVariableOp2:
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
HidA3/MatMul/ReadVariableOpHidA3/MatMul/ReadVariableOp2:
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
:ÿÿÿÿÿÿÿÿÿßÁ
 
_user_specified_nameinputs
è
ú
A__inference_ConC2_layer_call_and_return_conditional_losses_184051

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿG>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>
 
_user_specified_nameinputs
á
F
*__inference_dropout_9_layer_call_fn_184330

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_1811662
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ":W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
 
_user_specified_nameinputs
è
ú
A__inference_ConA2_layer_call_and_return_conditional_losses_183884

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿI?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?
 
_user_specified_nameinputs

c
E__inference_dropout_3_layer_call_and_return_conditional_losses_184340

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG=:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=
 
_user_specified_nameinputs


%__inference_ConA_layer_call_fn_183739

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ConA_layer_call_and_return_conditional_losses_1809242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿßÁ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿßÁ
 
_user_specified_nameinputs
í
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_183843

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿH?:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?
 
_user_specified_nameinputs
¢
^
B__inference_PoolA3_layer_call_and_return_conditional_losses_180788

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

c
E__inference_dropout_9_layer_call_and_return_conditional_losses_184313

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ":W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
 
_user_specified_nameinputs
Õ
F
*__inference_flatten_9_layer_call_fn_184499

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ð* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_9_layer_call_and_return_conditional_losses_1812022
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ð2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿD;:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;
 
_user_specified_nameinputs
Æ
B
&__inference_PoolA_layer_call_fn_183774

inputs
identityå
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolA_layer_call_and_return_conditional_losses_1806782
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

õ
@__inference_HidB_layer_call_and_return_conditional_losses_181338

inputs3
matmul_readvariableop_resource: ©.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
: ©*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ©: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©
 
_user_specified_nameinputs

c
E__inference_dropout_6_layer_call_and_return_conditional_losses_181187

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
ó
ù
@__inference_ConB_layer_call_and_return_conditional_losses_180907

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿßÁ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿßÁ
 
_user_specified_nameinputs
í
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_184352

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG=:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=
 
_user_specified_nameinputs
Û
C
'__inference_PoolB3_layer_call_fn_184140

inputs
identityË
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolB3_layer_call_and_return_conditional_losses_1811392
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ!(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(
 
_user_specified_nameinputs
¸
]
A__inference_PoolC_layer_call_and_return_conditional_losses_183923

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ×»:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»
 
_user_specified_nameinputs

c
E__inference_dropout_3_layer_call_and_return_conditional_losses_181159

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG=:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=
 
_user_specified_nameinputs

õ
A__inference_HidB3_layer_call_and_return_conditional_losses_181372

inputs2
matmul_readvariableop_resource:
F.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
F*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿF: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
Ä
c
*__inference_dropout_9_layer_call_fn_184335

inputs
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_1819272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ"22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
 
_user_specified_nameinputs
µ
^
B__inference_PoolB3_layer_call_and_return_conditional_losses_184130

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ!(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(
 
_user_specified_nameinputs
á
F
*__inference_dropout_6_layer_call_fn_184249

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_1811872
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
¸
]
A__inference_PoolB_layer_call_and_return_conditional_losses_183789

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÙ½:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½
 
_user_specified_nameinputs
é
a
E__inference_flatten_8_layer_call_and_return_conditional_losses_181210

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ0 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿá2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿá2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG=:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=
 
_user_specified_nameinputs
ç
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_184417

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ #  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
è
ú
A__inference_ConB3_layer_call_and_return_conditional_losses_181094

inputs8
conv2d_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
í
d
E__inference_dropout_6_layer_call_and_return_conditional_losses_181858

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
¡
]
A__inference_PoolB_layer_call_and_return_conditional_losses_180700

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
F
*__inference_dropout_3_layer_call_fn_184357

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_1811592
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG=:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=
 
_user_specified_nameinputs
¸
]
A__inference_PoolE_layer_call_and_return_conditional_losses_181121

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÍ³:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³
 
_user_specified_nameinputs
ç
_
C__inference_flatten_layer_call_and_return_conditional_losses_181274

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÊk  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ×2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ×2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿI?:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?
 
_user_specified_nameinputs
Ä
c
*__inference_dropout_3_layer_call_fn_184362

inputs
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_1819502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG=22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=
 
_user_specified_nameinputs
Ö£
$
!__inference__wrapped_model_180669	
inputC
)model_conb_conv2d_readvariableop_resource:8
*model_conb_biasadd_readvariableop_resource:C
)model_cona_conv2d_readvariableop_resource:8
*model_cona_biasadd_readvariableop_resource:C
)model_conc_conv2d_readvariableop_resource:	8
*model_conc_biasadd_readvariableop_resource:D
*model_conb2_conv2d_readvariableop_resource:9
+model_conb2_biasadd_readvariableop_resource:D
*model_cona2_conv2d_readvariableop_resource:9
+model_cona2_biasadd_readvariableop_resource:C
)model_cone_conv2d_readvariableop_resource:8
*model_cone_biasadd_readvariableop_resource:C
)model_cond_conv2d_readvariableop_resource:	8
*model_cond_biasadd_readvariableop_resource:D
*model_conc2_conv2d_readvariableop_resource:9
+model_conc2_biasadd_readvariableop_resource:D
*model_conb3_conv2d_readvariableop_resource:(9
+model_conb3_biasadd_readvariableop_resource:(D
*model_cona3_conv2d_readvariableop_resource:(9
+model_cona3_biasadd_readvariableop_resource:(>
)model_hida_matmul_readvariableop_resource:Ê×9
*model_hida_biasadd_readvariableop_resource:	?
*model_hida2_matmul_readvariableop_resource:¤:
+model_hida2_biasadd_readvariableop_resource:	>
*model_hida3_matmul_readvariableop_resource:
F:
+model_hida3_biasadd_readvariableop_resource:	>
)model_hidb_matmul_readvariableop_resource: ©9
*model_hidb_biasadd_readvariableop_resource:	?
*model_hidb2_matmul_readvariableop_resource:¤:
+model_hidb2_biasadd_readvariableop_resource:	>
*model_hidb3_matmul_readvariableop_resource:
F:
+model_hidb3_biasadd_readvariableop_resource:	>
)model_hidc_matmul_readvariableop_resource:î9
*model_hidc_biasadd_readvariableop_resource:	?
*model_hidc2_matmul_readvariableop_resource:°:
+model_hidc2_biasadd_readvariableop_resource:	?
*model_hidd1_matmul_readvariableop_resource:á:
+model_hidd1_biasadd_readvariableop_resource:	?
*model_hide1_matmul_readvariableop_resource: ð:
+model_hide1_biasadd_readvariableop_resource:	=
)model_hid1_matmul_readvariableop_resource:
9
*model_hid1_biasadd_readvariableop_resource:	=
)model_hid2_matmul_readvariableop_resource:
9
*model_hid2_biasadd_readvariableop_resource:	>
+model_output_matmul_readvariableop_resource:	(:
,model_output_biasadd_readvariableop_resource:(
identity¢!model/ConA/BiasAdd/ReadVariableOp¢ model/ConA/Conv2D/ReadVariableOp¢"model/ConA2/BiasAdd/ReadVariableOp¢!model/ConA2/Conv2D/ReadVariableOp¢"model/ConA3/BiasAdd/ReadVariableOp¢!model/ConA3/Conv2D/ReadVariableOp¢!model/ConB/BiasAdd/ReadVariableOp¢ model/ConB/Conv2D/ReadVariableOp¢"model/ConB2/BiasAdd/ReadVariableOp¢!model/ConB2/Conv2D/ReadVariableOp¢"model/ConB3/BiasAdd/ReadVariableOp¢!model/ConB3/Conv2D/ReadVariableOp¢!model/ConC/BiasAdd/ReadVariableOp¢ model/ConC/Conv2D/ReadVariableOp¢"model/ConC2/BiasAdd/ReadVariableOp¢!model/ConC2/Conv2D/ReadVariableOp¢!model/ConD/BiasAdd/ReadVariableOp¢ model/ConD/Conv2D/ReadVariableOp¢!model/ConE/BiasAdd/ReadVariableOp¢ model/ConE/Conv2D/ReadVariableOp¢!model/Hid1/BiasAdd/ReadVariableOp¢ model/Hid1/MatMul/ReadVariableOp¢!model/Hid2/BiasAdd/ReadVariableOp¢ model/Hid2/MatMul/ReadVariableOp¢!model/HidA/BiasAdd/ReadVariableOp¢ model/HidA/MatMul/ReadVariableOp¢"model/HidA2/BiasAdd/ReadVariableOp¢!model/HidA2/MatMul/ReadVariableOp¢"model/HidA3/BiasAdd/ReadVariableOp¢!model/HidA3/MatMul/ReadVariableOp¢!model/HidB/BiasAdd/ReadVariableOp¢ model/HidB/MatMul/ReadVariableOp¢"model/HidB2/BiasAdd/ReadVariableOp¢!model/HidB2/MatMul/ReadVariableOp¢"model/HidB3/BiasAdd/ReadVariableOp¢!model/HidB3/MatMul/ReadVariableOp¢!model/HidC/BiasAdd/ReadVariableOp¢ model/HidC/MatMul/ReadVariableOp¢"model/HidC2/BiasAdd/ReadVariableOp¢!model/HidC2/MatMul/ReadVariableOp¢"model/HidD1/BiasAdd/ReadVariableOp¢!model/HidD1/MatMul/ReadVariableOp¢"model/HidE1/BiasAdd/ReadVariableOp¢!model/HidE1/MatMul/ReadVariableOp¢#model/output/BiasAdd/ReadVariableOp¢"model/output/MatMul/ReadVariableOp¶
 model/ConB/Conv2D/ReadVariableOpReadVariableOp)model_conb_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 model/ConB/Conv2D/ReadVariableOpÆ
model/ConB/Conv2DConv2Dinput(model/ConB/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½*
paddingVALID*
strides
2
model/ConB/Conv2D­
!model/ConB/BiasAdd/ReadVariableOpReadVariableOp*model_conb_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!model/ConB/BiasAdd/ReadVariableOp¶
model/ConB/BiasAddBiasAddmodel/ConB/Conv2D:output:0)model/ConB/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½2
model/ConB/BiasAdd
model/ConB/ReluRelumodel/ConB/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½2
model/ConB/Relu¶
 model/ConA/Conv2D/ReadVariableOpReadVariableOp)model_cona_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 model/ConA/Conv2D/ReadVariableOpÆ
model/ConA/Conv2DConv2Dinput(model/ConA/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿*
paddingVALID*
strides
2
model/ConA/Conv2D­
!model/ConA/BiasAdd/ReadVariableOpReadVariableOp*model_cona_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!model/ConA/BiasAdd/ReadVariableOp¶
model/ConA/BiasAddBiasAddmodel/ConA/Conv2D:output:0)model/ConA/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿2
model/ConA/BiasAdd
model/ConA/ReluRelumodel/ConA/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿2
model/ConA/ReluÁ
model/PoolB/MaxPoolMaxPoolmodel/ConB/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?*
ksize
*
paddingVALID*
strides
2
model/PoolB/MaxPoolÁ
model/PoolA/MaxPoolMaxPoolmodel/ConA/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?*
ksize
*
paddingVALID*
strides
2
model/PoolA/MaxPool¶
 model/ConC/Conv2D/ReadVariableOpReadVariableOp)model_conc_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02"
 model/ConC/Conv2D/ReadVariableOpÆ
model/ConC/Conv2DConv2Dinput(model/ConC/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»*
paddingVALID*
strides
2
model/ConC/Conv2D­
!model/ConC/BiasAdd/ReadVariableOpReadVariableOp*model_conc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!model/ConC/BiasAdd/ReadVariableOp¶
model/ConC/BiasAddBiasAddmodel/ConC/Conv2D:output:0)model/ConC/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»2
model/ConC/BiasAdd
model/ConC/ReluRelumodel/ConC/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»2
model/ConC/Relu
model/dropout_1/IdentityIdentitymodel/PoolB/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?2
model/dropout_1/Identity
model/dropout/IdentityIdentitymodel/PoolA/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?2
model/dropout/IdentityÁ
model/PoolC/MaxPoolMaxPoolmodel/ConC/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>*
ksize
*
paddingVALID*
strides
2
model/PoolC/MaxPool¹
!model/ConB2/Conv2D/ReadVariableOpReadVariableOp*model_conb2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02#
!model/ConB2/Conv2D/ReadVariableOpã
model/ConB2/Conv2DConv2D!model/dropout_1/Identity:output:0)model/ConB2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=*
paddingVALID*
strides
2
model/ConB2/Conv2D°
"model/ConB2/BiasAdd/ReadVariableOpReadVariableOp+model_conb2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/ConB2/BiasAdd/ReadVariableOp¸
model/ConB2/BiasAddBiasAddmodel/ConB2/Conv2D:output:0*model/ConB2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=2
model/ConB2/BiasAdd
model/ConB2/ReluRelumodel/ConB2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=2
model/ConB2/Relu¹
!model/ConA2/Conv2D/ReadVariableOpReadVariableOp*model_cona2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02#
!model/ConA2/Conv2D/ReadVariableOpá
model/ConA2/Conv2DConv2Dmodel/dropout/Identity:output:0)model/ConA2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=*
paddingVALID*
strides
2
model/ConA2/Conv2D°
"model/ConA2/BiasAdd/ReadVariableOpReadVariableOp+model_cona2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/ConA2/BiasAdd/ReadVariableOp¸
model/ConA2/BiasAddBiasAddmodel/ConA2/Conv2D:output:0*model/ConA2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2
model/ConA2/BiasAdd
model/ConA2/ReluRelumodel/ConA2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2
model/ConA2/Relu
model/dropout_2/IdentityIdentitymodel/PoolC/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>2
model/dropout_2/IdentityÄ
model/PoolB2/MaxPoolMaxPoolmodel/ConB2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
ksize
*
paddingVALID*
strides
2
model/PoolB2/MaxPoolÄ
model/PoolA2/MaxPoolMaxPoolmodel/ConA2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
ksize
*
paddingVALID*
strides
2
model/PoolA2/MaxPool¶
 model/ConE/Conv2D/ReadVariableOpReadVariableOp)model_cone_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 model/ConE/Conv2D/ReadVariableOpÆ
model/ConE/Conv2DConv2Dinput(model/ConE/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³*
paddingVALID*
strides
2
model/ConE/Conv2D­
!model/ConE/BiasAdd/ReadVariableOpReadVariableOp*model_cone_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!model/ConE/BiasAdd/ReadVariableOp¶
model/ConE/BiasAddBiasAddmodel/ConE/Conv2D:output:0)model/ConE/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³2
model/ConE/BiasAdd
model/ConE/ReluRelumodel/ConE/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³2
model/ConE/Relu¶
 model/ConD/Conv2D/ReadVariableOpReadVariableOp)model_cond_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02"
 model/ConD/Conv2D/ReadVariableOpÆ
model/ConD/Conv2DConv2Dinput(model/ConD/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹*
paddingVALID*
strides
2
model/ConD/Conv2D­
!model/ConD/BiasAdd/ReadVariableOpReadVariableOp*model_cond_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!model/ConD/BiasAdd/ReadVariableOp¶
model/ConD/BiasAddBiasAddmodel/ConD/Conv2D:output:0)model/ConD/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹2
model/ConD/BiasAdd
model/ConD/ReluRelumodel/ConD/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹2
model/ConD/Relu¹
!model/ConC2/Conv2D/ReadVariableOpReadVariableOp*model_conc2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02#
!model/ConC2/Conv2D/ReadVariableOpã
model/ConC2/Conv2DConv2D!model/dropout_2/Identity:output:0)model/ConC2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<*
paddingVALID*
strides
2
model/ConC2/Conv2D°
"model/ConC2/BiasAdd/ReadVariableOpReadVariableOp+model_conc2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/ConC2/BiasAdd/ReadVariableOp¸
model/ConC2/BiasAddBiasAddmodel/ConC2/Conv2D:output:0*model/ConC2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<2
model/ConC2/BiasAdd
model/ConC2/ReluRelumodel/ConC2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<2
model/ConC2/Relu¹
!model/ConB3/Conv2D/ReadVariableOpReadVariableOp*model_conb3_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02#
!model/ConB3/Conv2D/ReadVariableOpß
model/ConB3/Conv2DConv2Dmodel/PoolB2/MaxPool:output:0)model/ConB3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(*
paddingVALID*
strides
2
model/ConB3/Conv2D°
"model/ConB3/BiasAdd/ReadVariableOpReadVariableOp+model_conb3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02$
"model/ConB3/BiasAdd/ReadVariableOp¸
model/ConB3/BiasAddBiasAddmodel/ConB3/Conv2D:output:0*model/ConB3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2
model/ConB3/BiasAdd
model/ConB3/ReluRelumodel/ConB3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2
model/ConB3/Relu¹
!model/ConA3/Conv2D/ReadVariableOpReadVariableOp*model_cona3_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02#
!model/ConA3/Conv2D/ReadVariableOpß
model/ConA3/Conv2DConv2Dmodel/PoolA2/MaxPool:output:0)model/ConA3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(*
paddingVALID*
strides
2
model/ConA3/Conv2D°
"model/ConA3/BiasAdd/ReadVariableOpReadVariableOp+model_cona3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02$
"model/ConA3/BiasAdd/ReadVariableOp¸
model/ConA3/BiasAddBiasAddmodel/ConA3/Conv2D:output:0*model/ConA3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2
model/ConA3/BiasAdd
model/ConA3/ReluRelumodel/ConA3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2
model/ConA3/ReluÁ
model/PoolE/MaxPoolMaxPoolmodel/ConE/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;*
ksize
*
paddingVALID*
strides
2
model/PoolE/MaxPoolÁ
model/PoolD/MaxPoolMaxPoolmodel/ConD/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=*
ksize
*
paddingVALID*
strides
2
model/PoolD/MaxPoolÄ
model/PoolC2/MaxPoolMaxPoolmodel/ConC2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"*
ksize
*
paddingVALID*
strides
2
model/PoolC2/MaxPoolÄ
model/PoolB3/MaxPoolMaxPoolmodel/ConB3/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
ksize
*
paddingVALID*
strides
2
model/PoolB3/MaxPoolÄ
model/PoolA3/MaxPoolMaxPoolmodel/ConA3/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
ksize
*
paddingVALID*
strides
2
model/PoolA3/MaxPool
model/dropout_4/IdentityIdentitymodel/PoolE/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;2
model/dropout_4/Identity
model/dropout_3/IdentityIdentitymodel/PoolD/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2
model/dropout_3/Identity
model/dropout_9/IdentityIdentitymodel/PoolC2/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2
model/dropout_9/Identity
model/dropout_8/IdentityIdentitymodel/PoolB3/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
model/dropout_8/Identity
model/dropout_7/IdentityIdentitymodel/PoolB2/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
model/dropout_7/Identity
model/dropout_6/IdentityIdentitymodel/PoolA3/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
model/dropout_6/Identity
model/dropout_5/IdentityIdentitymodel/PoolA2/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
model/dropout_5/Identity
model/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ x 2
model/flatten_9/Const´
model/flatten_9/ReshapeReshape!model/dropout_4/Identity:output:0model/flatten_9/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ð2
model/flatten_9/Reshape
model/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ0 2
model/flatten_8/Const´
model/flatten_8/ReshapeReshape!model/dropout_3/Identity:output:0model/flatten_8/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿá2
model/flatten_8/Reshape
model/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ°O  2
model/flatten_7/Const´
model/flatten_7/ReshapeReshape!model/dropout_9/Identity:output:0model/flatten_7/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
model/flatten_7/Reshape
model/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿî 2
model/flatten_6/Const´
model/flatten_6/ReshapeReshape!model/dropout_2/Identity:output:0model/flatten_6/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿî2
model/flatten_6/Reshape
model/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ #  2
model/flatten_5/Const³
model/flatten_5/ReshapeReshape!model/dropout_8/Identity:output:0model/flatten_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
model/flatten_5/Reshape
model/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿR  2
model/flatten_4/Const´
model/flatten_4/ReshapeReshape!model/dropout_7/Identity:output:0model/flatten_4/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤2
model/flatten_4/Reshape
model/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ Ô  2
model/flatten_3/Const¯
model/flatten_3/ReshapeReshapemodel/PoolB/MaxPool:output:0model/flatten_3/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©2
model/flatten_3/Reshape
model/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ #  2
model/flatten_2/Const³
model/flatten_2/ReshapeReshape!model/dropout_6/Identity:output:0model/flatten_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
model/flatten_2/Reshape
model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿR  2
model/flatten_1/Const´
model/flatten_1/ReshapeReshape!model/dropout_5/Identity:output:0model/flatten_1/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤2
model/flatten_1/Reshape{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÊk  2
model/flatten/Const©
model/flatten/ReshapeReshapemodel/PoolA/MaxPool:output:0model/flatten/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ×2
model/flatten/Reshape±
 model/HidA/MatMul/ReadVariableOpReadVariableOp)model_hida_matmul_readvariableop_resource*!
_output_shapes
:Ê×*
dtype02"
 model/HidA/MatMul/ReadVariableOp­
model/HidA/MatMulMatMulmodel/flatten/Reshape:output:0(model/HidA/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidA/MatMul®
!model/HidA/BiasAdd/ReadVariableOpReadVariableOp*model_hida_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!model/HidA/BiasAdd/ReadVariableOp®
model/HidA/BiasAddBiasAddmodel/HidA/MatMul:product:0)model/HidA/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidA/BiasAddz
model/HidA/ReluRelumodel/HidA/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidA/Relu´
!model/HidA2/MatMul/ReadVariableOpReadVariableOp*model_hida2_matmul_readvariableop_resource*!
_output_shapes
:¤*
dtype02#
!model/HidA2/MatMul/ReadVariableOp²
model/HidA2/MatMulMatMul model/flatten_1/Reshape:output:0)model/HidA2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidA2/MatMul±
"model/HidA2/BiasAdd/ReadVariableOpReadVariableOp+model_hida2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"model/HidA2/BiasAdd/ReadVariableOp²
model/HidA2/BiasAddBiasAddmodel/HidA2/MatMul:product:0*model/HidA2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidA2/BiasAdd}
model/HidA2/ReluRelumodel/HidA2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidA2/Relu³
!model/HidA3/MatMul/ReadVariableOpReadVariableOp*model_hida3_matmul_readvariableop_resource* 
_output_shapes
:
F*
dtype02#
!model/HidA3/MatMul/ReadVariableOp²
model/HidA3/MatMulMatMul model/flatten_2/Reshape:output:0)model/HidA3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidA3/MatMul±
"model/HidA3/BiasAdd/ReadVariableOpReadVariableOp+model_hida3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"model/HidA3/BiasAdd/ReadVariableOp²
model/HidA3/BiasAddBiasAddmodel/HidA3/MatMul:product:0*model/HidA3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidA3/BiasAdd}
model/HidA3/ReluRelumodel/HidA3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidA3/Relu±
 model/HidB/MatMul/ReadVariableOpReadVariableOp)model_hidb_matmul_readvariableop_resource*!
_output_shapes
: ©*
dtype02"
 model/HidB/MatMul/ReadVariableOp¯
model/HidB/MatMulMatMul model/flatten_3/Reshape:output:0(model/HidB/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidB/MatMul®
!model/HidB/BiasAdd/ReadVariableOpReadVariableOp*model_hidb_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!model/HidB/BiasAdd/ReadVariableOp®
model/HidB/BiasAddBiasAddmodel/HidB/MatMul:product:0)model/HidB/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidB/BiasAddz
model/HidB/ReluRelumodel/HidB/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidB/Relu´
!model/HidB2/MatMul/ReadVariableOpReadVariableOp*model_hidb2_matmul_readvariableop_resource*!
_output_shapes
:¤*
dtype02#
!model/HidB2/MatMul/ReadVariableOp²
model/HidB2/MatMulMatMul model/flatten_4/Reshape:output:0)model/HidB2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidB2/MatMul±
"model/HidB2/BiasAdd/ReadVariableOpReadVariableOp+model_hidb2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"model/HidB2/BiasAdd/ReadVariableOp²
model/HidB2/BiasAddBiasAddmodel/HidB2/MatMul:product:0*model/HidB2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidB2/BiasAdd}
model/HidB2/ReluRelumodel/HidB2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidB2/Relu³
!model/HidB3/MatMul/ReadVariableOpReadVariableOp*model_hidb3_matmul_readvariableop_resource* 
_output_shapes
:
F*
dtype02#
!model/HidB3/MatMul/ReadVariableOp²
model/HidB3/MatMulMatMul model/flatten_5/Reshape:output:0)model/HidB3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidB3/MatMul±
"model/HidB3/BiasAdd/ReadVariableOpReadVariableOp+model_hidb3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"model/HidB3/BiasAdd/ReadVariableOp²
model/HidB3/BiasAddBiasAddmodel/HidB3/MatMul:product:0*model/HidB3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidB3/BiasAdd}
model/HidB3/ReluRelumodel/HidB3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidB3/Relu±
 model/HidC/MatMul/ReadVariableOpReadVariableOp)model_hidc_matmul_readvariableop_resource*!
_output_shapes
:î*
dtype02"
 model/HidC/MatMul/ReadVariableOp¯
model/HidC/MatMulMatMul model/flatten_6/Reshape:output:0(model/HidC/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidC/MatMul®
!model/HidC/BiasAdd/ReadVariableOpReadVariableOp*model_hidc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!model/HidC/BiasAdd/ReadVariableOp®
model/HidC/BiasAddBiasAddmodel/HidC/MatMul:product:0)model/HidC/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidC/BiasAddz
model/HidC/ReluRelumodel/HidC/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidC/Relu´
!model/HidC2/MatMul/ReadVariableOpReadVariableOp*model_hidc2_matmul_readvariableop_resource*!
_output_shapes
:°*
dtype02#
!model/HidC2/MatMul/ReadVariableOp²
model/HidC2/MatMulMatMul model/flatten_7/Reshape:output:0)model/HidC2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidC2/MatMul±
"model/HidC2/BiasAdd/ReadVariableOpReadVariableOp+model_hidc2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"model/HidC2/BiasAdd/ReadVariableOp²
model/HidC2/BiasAddBiasAddmodel/HidC2/MatMul:product:0*model/HidC2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidC2/BiasAdd}
model/HidC2/ReluRelumodel/HidC2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidC2/Relu´
!model/HidD1/MatMul/ReadVariableOpReadVariableOp*model_hidd1_matmul_readvariableop_resource*!
_output_shapes
:á*
dtype02#
!model/HidD1/MatMul/ReadVariableOp²
model/HidD1/MatMulMatMul model/flatten_8/Reshape:output:0)model/HidD1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidD1/MatMul±
"model/HidD1/BiasAdd/ReadVariableOpReadVariableOp+model_hidd1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"model/HidD1/BiasAdd/ReadVariableOp²
model/HidD1/BiasAddBiasAddmodel/HidD1/MatMul:product:0*model/HidD1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidD1/BiasAdd}
model/HidD1/ReluRelumodel/HidD1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidD1/Relu´
!model/HidE1/MatMul/ReadVariableOpReadVariableOp*model_hide1_matmul_readvariableop_resource*!
_output_shapes
: ð*
dtype02#
!model/HidE1/MatMul/ReadVariableOp²
model/HidE1/MatMulMatMul model/flatten_9/Reshape:output:0)model/HidE1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidE1/MatMul±
"model/HidE1/BiasAdd/ReadVariableOpReadVariableOp+model_hide1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"model/HidE1/BiasAdd/ReadVariableOp²
model/HidE1/BiasAddBiasAddmodel/HidE1/MatMul:product:0*model/HidE1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidE1/BiasAdd}
model/HidE1/ReluRelumodel/HidE1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/HidE1/Relu
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axisá
model/concatenate/concatConcatV2model/HidA/Relu:activations:0model/HidA2/Relu:activations:0model/HidA3/Relu:activations:0model/HidB/Relu:activations:0model/HidB2/Relu:activations:0model/HidB3/Relu:activations:0model/HidC/Relu:activations:0model/HidC2/Relu:activations:0model/HidD1/Relu:activations:0model/HidE1/Relu:activations:0&model/concatenate/concat/axis:output:0*
N
*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/concatenate/concat°
 model/Hid1/MatMul/ReadVariableOpReadVariableOp)model_hid1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 model/Hid1/MatMul/ReadVariableOp°
model/Hid1/MatMulMatMul!model/concatenate/concat:output:0(model/Hid1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/Hid1/MatMul®
!model/Hid1/BiasAdd/ReadVariableOpReadVariableOp*model_hid1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!model/Hid1/BiasAdd/ReadVariableOp®
model/Hid1/BiasAddBiasAddmodel/Hid1/MatMul:product:0)model/Hid1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/Hid1/BiasAddz
model/Hid1/ReluRelumodel/Hid1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/Hid1/Relu°
 model/Hid2/MatMul/ReadVariableOpReadVariableOp)model_hid2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 model/Hid2/MatMul/ReadVariableOp¬
model/Hid2/MatMulMatMulmodel/Hid1/Relu:activations:0(model/Hid2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/Hid2/MatMul®
!model/Hid2/BiasAdd/ReadVariableOpReadVariableOp*model_hid2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!model/Hid2/BiasAdd/ReadVariableOp®
model/Hid2/BiasAddBiasAddmodel/Hid2/MatMul:product:0)model/Hid2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/Hid2/BiasAddz
model/Hid2/TanhTanhmodel/Hid2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/Hid2/Tanhµ
"model/output/MatMul/ReadVariableOpReadVariableOp+model_output_matmul_readvariableop_resource*
_output_shapes
:	(*
dtype02$
"model/output/MatMul/ReadVariableOp§
model/output/MatMulMatMulmodel/Hid2/Tanh:y:0*model/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
model/output/MatMul³
#model/output/BiasAdd/ReadVariableOpReadVariableOp,model_output_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02%
#model/output/BiasAdd/ReadVariableOpµ
model/output/BiasAddBiasAddmodel/output/MatMul:product:0+model/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
model/output/BiasAdd
model/output/SoftmaxSoftmaxmodel/output/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
model/output/Softmaxy
IdentityIdentitymodel/output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

IdentityË
NoOpNoOp"^model/ConA/BiasAdd/ReadVariableOp!^model/ConA/Conv2D/ReadVariableOp#^model/ConA2/BiasAdd/ReadVariableOp"^model/ConA2/Conv2D/ReadVariableOp#^model/ConA3/BiasAdd/ReadVariableOp"^model/ConA3/Conv2D/ReadVariableOp"^model/ConB/BiasAdd/ReadVariableOp!^model/ConB/Conv2D/ReadVariableOp#^model/ConB2/BiasAdd/ReadVariableOp"^model/ConB2/Conv2D/ReadVariableOp#^model/ConB3/BiasAdd/ReadVariableOp"^model/ConB3/Conv2D/ReadVariableOp"^model/ConC/BiasAdd/ReadVariableOp!^model/ConC/Conv2D/ReadVariableOp#^model/ConC2/BiasAdd/ReadVariableOp"^model/ConC2/Conv2D/ReadVariableOp"^model/ConD/BiasAdd/ReadVariableOp!^model/ConD/Conv2D/ReadVariableOp"^model/ConE/BiasAdd/ReadVariableOp!^model/ConE/Conv2D/ReadVariableOp"^model/Hid1/BiasAdd/ReadVariableOp!^model/Hid1/MatMul/ReadVariableOp"^model/Hid2/BiasAdd/ReadVariableOp!^model/Hid2/MatMul/ReadVariableOp"^model/HidA/BiasAdd/ReadVariableOp!^model/HidA/MatMul/ReadVariableOp#^model/HidA2/BiasAdd/ReadVariableOp"^model/HidA2/MatMul/ReadVariableOp#^model/HidA3/BiasAdd/ReadVariableOp"^model/HidA3/MatMul/ReadVariableOp"^model/HidB/BiasAdd/ReadVariableOp!^model/HidB/MatMul/ReadVariableOp#^model/HidB2/BiasAdd/ReadVariableOp"^model/HidB2/MatMul/ReadVariableOp#^model/HidB3/BiasAdd/ReadVariableOp"^model/HidB3/MatMul/ReadVariableOp"^model/HidC/BiasAdd/ReadVariableOp!^model/HidC/MatMul/ReadVariableOp#^model/HidC2/BiasAdd/ReadVariableOp"^model/HidC2/MatMul/ReadVariableOp#^model/HidD1/BiasAdd/ReadVariableOp"^model/HidD1/MatMul/ReadVariableOp#^model/HidE1/BiasAdd/ReadVariableOp"^model/HidE1/MatMul/ReadVariableOp$^model/output/BiasAdd/ReadVariableOp#^model/output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes{
y:ÿÿÿÿÿÿÿÿÿßÁ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!model/ConA/BiasAdd/ReadVariableOp!model/ConA/BiasAdd/ReadVariableOp2D
 model/ConA/Conv2D/ReadVariableOp model/ConA/Conv2D/ReadVariableOp2H
"model/ConA2/BiasAdd/ReadVariableOp"model/ConA2/BiasAdd/ReadVariableOp2F
!model/ConA2/Conv2D/ReadVariableOp!model/ConA2/Conv2D/ReadVariableOp2H
"model/ConA3/BiasAdd/ReadVariableOp"model/ConA3/BiasAdd/ReadVariableOp2F
!model/ConA3/Conv2D/ReadVariableOp!model/ConA3/Conv2D/ReadVariableOp2F
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
!model/HidA3/MatMul/ReadVariableOp!model/HidA3/MatMul/ReadVariableOp2F
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
:ÿÿÿÿÿÿÿÿÿßÁ

_user_specified_nameinput
Æ
B
&__inference_PoolB_layer_call_fn_183794

inputs
identityå
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolB_layer_call_and_return_conditional_losses_1807002
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
F
*__inference_flatten_4_layer_call_fn_184444

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_1812422
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
û
Ö
&__inference_model_layer_call_fn_181610	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:	
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:	

unknown_12:$

unknown_13:

unknown_14:$

unknown_15:(

unknown_16:($

unknown_17:(

unknown_18:(

unknown_19:Ê×

unknown_20:	

unknown_21:¤

unknown_22:	

unknown_23:
F

unknown_24:	

unknown_25: ©

unknown_26:	

unknown_27:¤

unknown_28:	

unknown_29:
F

unknown_30:	

unknown_31:î

unknown_32:	

unknown_33:°

unknown_34:	

unknown_35:á

unknown_36:	

unknown_37: ð

unknown_38:	

unknown_39:


unknown_40:	

unknown_41:


unknown_42:	

unknown_43:	(

unknown_44:(
identity¢StatefulPartitionedCallÒ
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
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1815152
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes{
y:ÿÿÿÿÿÿÿÿÿßÁ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿßÁ

_user_specified_nameinput
á
F
*__inference_dropout_5_layer_call_fn_184222

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_1811942
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
Ñ
ÿ
A__inference_model_layer_call_and_return_conditional_losses_183248

inputs=
#conb_conv2d_readvariableop_resource:2
$conb_biasadd_readvariableop_resource:=
#cona_conv2d_readvariableop_resource:2
$cona_biasadd_readvariableop_resource:=
#conc_conv2d_readvariableop_resource:	2
$conc_biasadd_readvariableop_resource:>
$conb2_conv2d_readvariableop_resource:3
%conb2_biasadd_readvariableop_resource:>
$cona2_conv2d_readvariableop_resource:3
%cona2_biasadd_readvariableop_resource:=
#cone_conv2d_readvariableop_resource:2
$cone_biasadd_readvariableop_resource:=
#cond_conv2d_readvariableop_resource:	2
$cond_biasadd_readvariableop_resource:>
$conc2_conv2d_readvariableop_resource:3
%conc2_biasadd_readvariableop_resource:>
$conb3_conv2d_readvariableop_resource:(3
%conb3_biasadd_readvariableop_resource:(>
$cona3_conv2d_readvariableop_resource:(3
%cona3_biasadd_readvariableop_resource:(8
#hida_matmul_readvariableop_resource:Ê×3
$hida_biasadd_readvariableop_resource:	9
$hida2_matmul_readvariableop_resource:¤4
%hida2_biasadd_readvariableop_resource:	8
$hida3_matmul_readvariableop_resource:
F4
%hida3_biasadd_readvariableop_resource:	8
#hidb_matmul_readvariableop_resource: ©3
$hidb_biasadd_readvariableop_resource:	9
$hidb2_matmul_readvariableop_resource:¤4
%hidb2_biasadd_readvariableop_resource:	8
$hidb3_matmul_readvariableop_resource:
F4
%hidb3_biasadd_readvariableop_resource:	8
#hidc_matmul_readvariableop_resource:î3
$hidc_biasadd_readvariableop_resource:	9
$hidc2_matmul_readvariableop_resource:°4
%hidc2_biasadd_readvariableop_resource:	9
$hidd1_matmul_readvariableop_resource:á4
%hidd1_biasadd_readvariableop_resource:	9
$hide1_matmul_readvariableop_resource: ð4
%hide1_biasadd_readvariableop_resource:	7
#hid1_matmul_readvariableop_resource:
3
$hid1_biasadd_readvariableop_resource:	7
#hid2_matmul_readvariableop_resource:
3
$hid2_biasadd_readvariableop_resource:	8
%output_matmul_readvariableop_resource:	(4
&output_biasadd_readvariableop_resource:(
identity¢ConA/BiasAdd/ReadVariableOp¢ConA/Conv2D/ReadVariableOp¢ConA2/BiasAdd/ReadVariableOp¢ConA2/Conv2D/ReadVariableOp¢ConA3/BiasAdd/ReadVariableOp¢ConA3/Conv2D/ReadVariableOp¢ConB/BiasAdd/ReadVariableOp¢ConB/Conv2D/ReadVariableOp¢ConB2/BiasAdd/ReadVariableOp¢ConB2/Conv2D/ReadVariableOp¢ConB3/BiasAdd/ReadVariableOp¢ConB3/Conv2D/ReadVariableOp¢ConC/BiasAdd/ReadVariableOp¢ConC/Conv2D/ReadVariableOp¢ConC2/BiasAdd/ReadVariableOp¢ConC2/Conv2D/ReadVariableOp¢ConD/BiasAdd/ReadVariableOp¢ConD/Conv2D/ReadVariableOp¢ConE/BiasAdd/ReadVariableOp¢ConE/Conv2D/ReadVariableOp¢Hid1/BiasAdd/ReadVariableOp¢Hid1/MatMul/ReadVariableOp¢Hid2/BiasAdd/ReadVariableOp¢Hid2/MatMul/ReadVariableOp¢HidA/BiasAdd/ReadVariableOp¢HidA/MatMul/ReadVariableOp¢HidA2/BiasAdd/ReadVariableOp¢HidA2/MatMul/ReadVariableOp¢HidA3/BiasAdd/ReadVariableOp¢HidA3/MatMul/ReadVariableOp¢HidB/BiasAdd/ReadVariableOp¢HidB/MatMul/ReadVariableOp¢HidB2/BiasAdd/ReadVariableOp¢HidB2/MatMul/ReadVariableOp¢HidB3/BiasAdd/ReadVariableOp¢HidB3/MatMul/ReadVariableOp¢HidC/BiasAdd/ReadVariableOp¢HidC/MatMul/ReadVariableOp¢HidC2/BiasAdd/ReadVariableOp¢HidC2/MatMul/ReadVariableOp¢HidD1/BiasAdd/ReadVariableOp¢HidD1/MatMul/ReadVariableOp¢HidE1/BiasAdd/ReadVariableOp¢HidE1/MatMul/ReadVariableOp¢output/BiasAdd/ReadVariableOp¢output/MatMul/ReadVariableOp¤
ConB/Conv2D/ReadVariableOpReadVariableOp#conb_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConB/Conv2D/ReadVariableOpµ
ConB/Conv2DConv2Dinputs"ConB/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½*
paddingVALID*
strides
2
ConB/Conv2D
ConB/BiasAdd/ReadVariableOpReadVariableOp$conb_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConB/BiasAdd/ReadVariableOp
ConB/BiasAddBiasAddConB/Conv2D:output:0#ConB/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½2
ConB/BiasAddq
	ConB/ReluReluConB/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½2
	ConB/Relu¤
ConA/Conv2D/ReadVariableOpReadVariableOp#cona_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConA/Conv2D/ReadVariableOpµ
ConA/Conv2DConv2Dinputs"ConA/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿*
paddingVALID*
strides
2
ConA/Conv2D
ConA/BiasAdd/ReadVariableOpReadVariableOp$cona_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConA/BiasAdd/ReadVariableOp
ConA/BiasAddBiasAddConA/Conv2D:output:0#ConA/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿2
ConA/BiasAddq
	ConA/ReluReluConA/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿2
	ConA/Relu¯
PoolB/MaxPoolMaxPoolConB/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?*
ksize
*
paddingVALID*
strides
2
PoolB/MaxPool¯
PoolA/MaxPoolMaxPoolConA/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?*
ksize
*
paddingVALID*
strides
2
PoolA/MaxPool¤
ConC/Conv2D/ReadVariableOpReadVariableOp#conc_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
ConC/Conv2D/ReadVariableOpµ
ConC/Conv2DConv2Dinputs"ConC/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»*
paddingVALID*
strides
2
ConC/Conv2D
ConC/BiasAdd/ReadVariableOpReadVariableOp$conc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConC/BiasAdd/ReadVariableOp
ConC/BiasAddBiasAddConC/Conv2D:output:0#ConC/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»2
ConC/BiasAddq
	ConC/ReluReluConC/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»2
	ConC/Relu
dropout_1/IdentityIdentityPoolB/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?2
dropout_1/Identity
dropout/IdentityIdentityPoolA/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?2
dropout/Identity¯
PoolC/MaxPoolMaxPoolConC/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>*
ksize
*
paddingVALID*
strides
2
PoolC/MaxPool§
ConB2/Conv2D/ReadVariableOpReadVariableOp$conb2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConB2/Conv2D/ReadVariableOpË
ConB2/Conv2DConv2Ddropout_1/Identity:output:0#ConB2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=*
paddingVALID*
strides
2
ConB2/Conv2D
ConB2/BiasAdd/ReadVariableOpReadVariableOp%conb2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConB2/BiasAdd/ReadVariableOp 
ConB2/BiasAddBiasAddConB2/Conv2D:output:0$ConB2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=2
ConB2/BiasAddr

ConB2/ReluReluConB2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=2

ConB2/Relu§
ConA2/Conv2D/ReadVariableOpReadVariableOp$cona2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConA2/Conv2D/ReadVariableOpÉ
ConA2/Conv2DConv2Ddropout/Identity:output:0#ConA2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=*
paddingVALID*
strides
2
ConA2/Conv2D
ConA2/BiasAdd/ReadVariableOpReadVariableOp%cona2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConA2/BiasAdd/ReadVariableOp 
ConA2/BiasAddBiasAddConA2/Conv2D:output:0$ConA2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2
ConA2/BiasAddr

ConA2/ReluReluConA2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2

ConA2/Relu
dropout_2/IdentityIdentityPoolC/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>2
dropout_2/Identity²
PoolB2/MaxPoolMaxPoolConB2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
ksize
*
paddingVALID*
strides
2
PoolB2/MaxPool²
PoolA2/MaxPoolMaxPoolConA2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
ksize
*
paddingVALID*
strides
2
PoolA2/MaxPool¤
ConE/Conv2D/ReadVariableOpReadVariableOp#cone_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConE/Conv2D/ReadVariableOpµ
ConE/Conv2DConv2Dinputs"ConE/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³*
paddingVALID*
strides
2
ConE/Conv2D
ConE/BiasAdd/ReadVariableOpReadVariableOp$cone_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConE/BiasAdd/ReadVariableOp
ConE/BiasAddBiasAddConE/Conv2D:output:0#ConE/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³2
ConE/BiasAddq
	ConE/ReluReluConE/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³2
	ConE/Relu¤
ConD/Conv2D/ReadVariableOpReadVariableOp#cond_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
ConD/Conv2D/ReadVariableOpµ
ConD/Conv2DConv2Dinputs"ConD/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹*
paddingVALID*
strides
2
ConD/Conv2D
ConD/BiasAdd/ReadVariableOpReadVariableOp$cond_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConD/BiasAdd/ReadVariableOp
ConD/BiasAddBiasAddConD/Conv2D:output:0#ConD/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹2
ConD/BiasAddq
	ConD/ReluReluConD/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹2
	ConD/Relu§
ConC2/Conv2D/ReadVariableOpReadVariableOp$conc2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConC2/Conv2D/ReadVariableOpË
ConC2/Conv2DConv2Ddropout_2/Identity:output:0#ConC2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<*
paddingVALID*
strides
2
ConC2/Conv2D
ConC2/BiasAdd/ReadVariableOpReadVariableOp%conc2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConC2/BiasAdd/ReadVariableOp 
ConC2/BiasAddBiasAddConC2/Conv2D:output:0$ConC2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<2
ConC2/BiasAddr

ConC2/ReluReluConC2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<2

ConC2/Relu§
ConB3/Conv2D/ReadVariableOpReadVariableOp$conb3_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
ConB3/Conv2D/ReadVariableOpÇ
ConB3/Conv2DConv2DPoolB2/MaxPool:output:0#ConB3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(*
paddingVALID*
strides
2
ConB3/Conv2D
ConB3/BiasAdd/ReadVariableOpReadVariableOp%conb3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
ConB3/BiasAdd/ReadVariableOp 
ConB3/BiasAddBiasAddConB3/Conv2D:output:0$ConB3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2
ConB3/BiasAddr

ConB3/ReluReluConB3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2

ConB3/Relu§
ConA3/Conv2D/ReadVariableOpReadVariableOp$cona3_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
ConA3/Conv2D/ReadVariableOpÇ
ConA3/Conv2DConv2DPoolA2/MaxPool:output:0#ConA3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(*
paddingVALID*
strides
2
ConA3/Conv2D
ConA3/BiasAdd/ReadVariableOpReadVariableOp%cona3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
ConA3/BiasAdd/ReadVariableOp 
ConA3/BiasAddBiasAddConA3/Conv2D:output:0$ConA3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2
ConA3/BiasAddr

ConA3/ReluReluConA3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2

ConA3/Relu¯
PoolE/MaxPoolMaxPoolConE/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;*
ksize
*
paddingVALID*
strides
2
PoolE/MaxPool¯
PoolD/MaxPoolMaxPoolConD/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=*
ksize
*
paddingVALID*
strides
2
PoolD/MaxPool²
PoolC2/MaxPoolMaxPoolConC2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"*
ksize
*
paddingVALID*
strides
2
PoolC2/MaxPool²
PoolB3/MaxPoolMaxPoolConB3/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
ksize
*
paddingVALID*
strides
2
PoolB3/MaxPool²
PoolA3/MaxPoolMaxPoolConA3/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
ksize
*
paddingVALID*
strides
2
PoolA3/MaxPool
dropout_4/IdentityIdentityPoolE/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;2
dropout_4/Identity
dropout_3/IdentityIdentityPoolD/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2
dropout_3/Identity
dropout_9/IdentityIdentityPoolC2/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2
dropout_9/Identity
dropout_8/IdentityIdentityPoolB3/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dropout_8/Identity
dropout_7/IdentityIdentityPoolB2/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
dropout_7/Identity
dropout_6/IdentityIdentityPoolA3/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dropout_6/Identity
dropout_5/IdentityIdentityPoolA2/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
dropout_5/Identitys
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ x 2
flatten_9/Const
flatten_9/ReshapeReshapedropout_4/Identity:output:0flatten_9/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ð2
flatten_9/Reshapes
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ0 2
flatten_8/Const
flatten_8/ReshapeReshapedropout_3/Identity:output:0flatten_8/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿá2
flatten_8/Reshapes
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ°O  2
flatten_7/Const
flatten_7/ReshapeReshapedropout_9/Identity:output:0flatten_7/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
flatten_7/Reshapes
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿî 2
flatten_6/Const
flatten_6/ReshapeReshapedropout_2/Identity:output:0flatten_6/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿî2
flatten_6/Reshapes
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ #  2
flatten_5/Const
flatten_5/ReshapeReshapedropout_8/Identity:output:0flatten_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
flatten_5/Reshapes
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿR  2
flatten_4/Const
flatten_4/ReshapeReshapedropout_7/Identity:output:0flatten_4/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤2
flatten_4/Reshapes
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ Ô  2
flatten_3/Const
flatten_3/ReshapeReshapePoolB/MaxPool:output:0flatten_3/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©2
flatten_3/Reshapes
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ #  2
flatten_2/Const
flatten_2/ReshapeReshapedropout_6/Identity:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
flatten_2/Reshapes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿR  2
flatten_1/Const
flatten_1/ReshapeReshapedropout_5/Identity:output:0flatten_1/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤2
flatten_1/Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÊk  2
flatten/Const
flatten/ReshapeReshapePoolA/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ×2
flatten/Reshape
HidA/MatMul/ReadVariableOpReadVariableOp#hida_matmul_readvariableop_resource*!
_output_shapes
:Ê×*
dtype02
HidA/MatMul/ReadVariableOp
HidA/MatMulMatMulflatten/Reshape:output:0"HidA/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidA/MatMul
HidA/BiasAdd/ReadVariableOpReadVariableOp$hida_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
HidA/BiasAdd/ReadVariableOp
HidA/BiasAddBiasAddHidA/MatMul:product:0#HidA/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidA/BiasAddh
	HidA/ReluReluHidA/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	HidA/Relu¢
HidA2/MatMul/ReadVariableOpReadVariableOp$hida2_matmul_readvariableop_resource*!
_output_shapes
:¤*
dtype02
HidA2/MatMul/ReadVariableOp
HidA2/MatMulMatMulflatten_1/Reshape:output:0#HidA2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidA2/MatMul
HidA2/BiasAdd/ReadVariableOpReadVariableOp%hida2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
HidA2/BiasAdd/ReadVariableOp
HidA2/BiasAddBiasAddHidA2/MatMul:product:0$HidA2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidA2/BiasAddk

HidA2/ReluReluHidA2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

HidA2/Relu¡
HidA3/MatMul/ReadVariableOpReadVariableOp$hida3_matmul_readvariableop_resource* 
_output_shapes
:
F*
dtype02
HidA3/MatMul/ReadVariableOp
HidA3/MatMulMatMulflatten_2/Reshape:output:0#HidA3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidA3/MatMul
HidA3/BiasAdd/ReadVariableOpReadVariableOp%hida3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
HidA3/BiasAdd/ReadVariableOp
HidA3/BiasAddBiasAddHidA3/MatMul:product:0$HidA3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidA3/BiasAddk

HidA3/ReluReluHidA3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

HidA3/Relu
HidB/MatMul/ReadVariableOpReadVariableOp#hidb_matmul_readvariableop_resource*!
_output_shapes
: ©*
dtype02
HidB/MatMul/ReadVariableOp
HidB/MatMulMatMulflatten_3/Reshape:output:0"HidB/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidB/MatMul
HidB/BiasAdd/ReadVariableOpReadVariableOp$hidb_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
HidB/BiasAdd/ReadVariableOp
HidB/BiasAddBiasAddHidB/MatMul:product:0#HidB/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidB/BiasAddh
	HidB/ReluReluHidB/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	HidB/Relu¢
HidB2/MatMul/ReadVariableOpReadVariableOp$hidb2_matmul_readvariableop_resource*!
_output_shapes
:¤*
dtype02
HidB2/MatMul/ReadVariableOp
HidB2/MatMulMatMulflatten_4/Reshape:output:0#HidB2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidB2/MatMul
HidB2/BiasAdd/ReadVariableOpReadVariableOp%hidb2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
HidB2/BiasAdd/ReadVariableOp
HidB2/BiasAddBiasAddHidB2/MatMul:product:0$HidB2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidB2/BiasAddk

HidB2/ReluReluHidB2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

HidB2/Relu¡
HidB3/MatMul/ReadVariableOpReadVariableOp$hidb3_matmul_readvariableop_resource* 
_output_shapes
:
F*
dtype02
HidB3/MatMul/ReadVariableOp
HidB3/MatMulMatMulflatten_5/Reshape:output:0#HidB3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidB3/MatMul
HidB3/BiasAdd/ReadVariableOpReadVariableOp%hidb3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
HidB3/BiasAdd/ReadVariableOp
HidB3/BiasAddBiasAddHidB3/MatMul:product:0$HidB3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidB3/BiasAddk

HidB3/ReluReluHidB3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

HidB3/Relu
HidC/MatMul/ReadVariableOpReadVariableOp#hidc_matmul_readvariableop_resource*!
_output_shapes
:î*
dtype02
HidC/MatMul/ReadVariableOp
HidC/MatMulMatMulflatten_6/Reshape:output:0"HidC/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidC/MatMul
HidC/BiasAdd/ReadVariableOpReadVariableOp$hidc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
HidC/BiasAdd/ReadVariableOp
HidC/BiasAddBiasAddHidC/MatMul:product:0#HidC/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidC/BiasAddh
	HidC/ReluReluHidC/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	HidC/Relu¢
HidC2/MatMul/ReadVariableOpReadVariableOp$hidc2_matmul_readvariableop_resource*!
_output_shapes
:°*
dtype02
HidC2/MatMul/ReadVariableOp
HidC2/MatMulMatMulflatten_7/Reshape:output:0#HidC2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidC2/MatMul
HidC2/BiasAdd/ReadVariableOpReadVariableOp%hidc2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
HidC2/BiasAdd/ReadVariableOp
HidC2/BiasAddBiasAddHidC2/MatMul:product:0$HidC2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidC2/BiasAddk

HidC2/ReluReluHidC2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

HidC2/Relu¢
HidD1/MatMul/ReadVariableOpReadVariableOp$hidd1_matmul_readvariableop_resource*!
_output_shapes
:á*
dtype02
HidD1/MatMul/ReadVariableOp
HidD1/MatMulMatMulflatten_8/Reshape:output:0#HidD1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidD1/MatMul
HidD1/BiasAdd/ReadVariableOpReadVariableOp%hidd1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
HidD1/BiasAdd/ReadVariableOp
HidD1/BiasAddBiasAddHidD1/MatMul:product:0$HidD1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidD1/BiasAddk

HidD1/ReluReluHidD1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

HidD1/Relu¢
HidE1/MatMul/ReadVariableOpReadVariableOp$hide1_matmul_readvariableop_resource*!
_output_shapes
: ð*
dtype02
HidE1/MatMul/ReadVariableOp
HidE1/MatMulMatMulflatten_9/Reshape:output:0#HidE1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidE1/MatMul
HidE1/BiasAdd/ReadVariableOpReadVariableOp%hide1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
HidE1/BiasAdd/ReadVariableOp
HidE1/BiasAddBiasAddHidE1/MatMul:product:0$HidE1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HidE1/BiasAddk

HidE1/ReluReluHidE1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

HidE1/Relut
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis
concatenate/concatConcatV2HidA/Relu:activations:0HidA2/Relu:activations:0HidA3/Relu:activations:0HidB/Relu:activations:0HidB2/Relu:activations:0HidB3/Relu:activations:0HidC/Relu:activations:0HidC2/Relu:activations:0HidD1/Relu:activations:0HidE1/Relu:activations:0 concatenate/concat/axis:output:0*
N
*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatenate/concat
Hid1/MatMul/ReadVariableOpReadVariableOp#hid1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
Hid1/MatMul/ReadVariableOp
Hid1/MatMulMatMulconcatenate/concat:output:0"Hid1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Hid1/MatMul
Hid1/BiasAdd/ReadVariableOpReadVariableOp$hid1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
Hid1/BiasAdd/ReadVariableOp
Hid1/BiasAddBiasAddHid1/MatMul:product:0#Hid1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Hid1/BiasAddh
	Hid1/ReluReluHid1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Hid1/Relu
Hid2/MatMul/ReadVariableOpReadVariableOp#hid2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
Hid2/MatMul/ReadVariableOp
Hid2/MatMulMatMulHid1/Relu:activations:0"Hid2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Hid2/MatMul
Hid2/BiasAdd/ReadVariableOpReadVariableOp$hid2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
Hid2/BiasAdd/ReadVariableOp
Hid2/BiasAddBiasAddHid2/MatMul:product:0#Hid2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Hid2/BiasAddh
	Hid2/TanhTanhHid2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Hid2/Tanh£
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	(*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMulHid2/Tanh:y:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
output/MatMul¡
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
output/Softmaxs
IdentityIdentityoutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity·
NoOpNoOp^ConA/BiasAdd/ReadVariableOp^ConA/Conv2D/ReadVariableOp^ConA2/BiasAdd/ReadVariableOp^ConA2/Conv2D/ReadVariableOp^ConA3/BiasAdd/ReadVariableOp^ConA3/Conv2D/ReadVariableOp^ConB/BiasAdd/ReadVariableOp^ConB/Conv2D/ReadVariableOp^ConB2/BiasAdd/ReadVariableOp^ConB2/Conv2D/ReadVariableOp^ConB3/BiasAdd/ReadVariableOp^ConB3/Conv2D/ReadVariableOp^ConC/BiasAdd/ReadVariableOp^ConC/Conv2D/ReadVariableOp^ConC2/BiasAdd/ReadVariableOp^ConC2/Conv2D/ReadVariableOp^ConD/BiasAdd/ReadVariableOp^ConD/Conv2D/ReadVariableOp^ConE/BiasAdd/ReadVariableOp^ConE/Conv2D/ReadVariableOp^Hid1/BiasAdd/ReadVariableOp^Hid1/MatMul/ReadVariableOp^Hid2/BiasAdd/ReadVariableOp^Hid2/MatMul/ReadVariableOp^HidA/BiasAdd/ReadVariableOp^HidA/MatMul/ReadVariableOp^HidA2/BiasAdd/ReadVariableOp^HidA2/MatMul/ReadVariableOp^HidA3/BiasAdd/ReadVariableOp^HidA3/MatMul/ReadVariableOp^HidB/BiasAdd/ReadVariableOp^HidB/MatMul/ReadVariableOp^HidB2/BiasAdd/ReadVariableOp^HidB2/MatMul/ReadVariableOp^HidB3/BiasAdd/ReadVariableOp^HidB3/MatMul/ReadVariableOp^HidC/BiasAdd/ReadVariableOp^HidC/MatMul/ReadVariableOp^HidC2/BiasAdd/ReadVariableOp^HidC2/MatMul/ReadVariableOp^HidD1/BiasAdd/ReadVariableOp^HidD1/MatMul/ReadVariableOp^HidE1/BiasAdd/ReadVariableOp^HidE1/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes{
y:ÿÿÿÿÿÿÿÿÿßÁ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
ConA/BiasAdd/ReadVariableOpConA/BiasAdd/ReadVariableOp28
ConA/Conv2D/ReadVariableOpConA/Conv2D/ReadVariableOp2<
ConA2/BiasAdd/ReadVariableOpConA2/BiasAdd/ReadVariableOp2:
ConA2/Conv2D/ReadVariableOpConA2/Conv2D/ReadVariableOp2<
ConA3/BiasAdd/ReadVariableOpConA3/BiasAdd/ReadVariableOp2:
ConA3/Conv2D/ReadVariableOpConA3/Conv2D/ReadVariableOp2:
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
HidA3/MatMul/ReadVariableOpHidA3/MatMul/ReadVariableOp2:
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
:ÿÿÿÿÿÿÿÿÿßÁ
 
_user_specified_nameinputs
í
d
E__inference_dropout_9_layer_call_and_return_conditional_losses_181927

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ":W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
 
_user_specified_nameinputs
ö

%__inference_HidB_layer_call_fn_184579

inputs
unknown: ©
	unknown_0:	
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_HidB_layer_call_and_return_conditional_losses_1813382
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ©: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©
 
_user_specified_nameinputs
á
F
*__inference_dropout_4_layer_call_fn_184384

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1811522
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿD;:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;
 
_user_specified_nameinputs
Ä
c
*__inference_dropout_7_layer_call_fn_184281

inputs
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_1818812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
Ý
D
(__inference_dropout_layer_call_fn_183821

inputs
identityÌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1809712
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿI?:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?
 
_user_specified_nameinputs

a
C__inference_dropout_layer_call_and_return_conditional_losses_183804

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿI?:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?
 
_user_specified_nameinputs

ö
A__inference_HidE1_layer_call_and_return_conditional_losses_184690

inputs3
matmul_readvariableop_resource: ð.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
: ð*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ð: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ð
 
_user_specified_nameinputs
ç
a
E__inference_flatten_5_layer_call_and_return_conditional_losses_181234

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ #  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
û
Ö
&__inference_model_layer_call_fn_182636	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:	
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:	

unknown_12:$

unknown_13:

unknown_14:$

unknown_15:(

unknown_16:($

unknown_17:(

unknown_18:(

unknown_19:Ê×

unknown_20:	

unknown_21:¤

unknown_22:	

unknown_23:
F

unknown_24:	

unknown_25: ©

unknown_26:	

unknown_27:¤

unknown_28:	

unknown_29:
F

unknown_30:	

unknown_31:î

unknown_32:	

unknown_33:°

unknown_34:	

unknown_35:á

unknown_36:	

unknown_37: ð

unknown_38:	

unknown_39:


unknown_40:	

unknown_41:


unknown_42:	

unknown_43:	(

unknown_44:(
identity¢StatefulPartitionedCallÒ
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
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1824442
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes{
y:ÿÿÿÿÿÿÿÿÿßÁ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿßÁ

_user_specified_nameinput
ç
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_181258

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ #  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
í
d
E__inference_dropout_5_layer_call_and_return_conditional_losses_181835

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs

ô
B__inference_output_layer_call_and_return_conditional_losses_184779

inputs1
matmul_readvariableop_resource:	(-
biasadd_readvariableop_resource:(
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ

&__inference_HidA3_layer_call_fn_184559

inputs
unknown:
F
	unknown_0:	
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidA3_layer_call_and_return_conditional_losses_1813212
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿF: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
þ
×
&__inference_model_layer_call_fn_183622

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:	
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:	

unknown_12:$

unknown_13:

unknown_14:$

unknown_15:(

unknown_16:($

unknown_17:(

unknown_18:(

unknown_19:Ê×

unknown_20:	

unknown_21:¤

unknown_22:	

unknown_23:
F

unknown_24:	

unknown_25: ©

unknown_26:	

unknown_27:¤

unknown_28:	

unknown_29:
F

unknown_30:	

unknown_31:î

unknown_32:	

unknown_33:°

unknown_34:	

unknown_35:á

unknown_36:	

unknown_37: ð

unknown_38:	

unknown_39:


unknown_40:	

unknown_41:


unknown_42:	

unknown_43:	(

unknown_44:(
identity¢StatefulPartitionedCallÓ
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
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1815152
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes{
y:ÿÿÿÿÿÿÿÿÿßÁ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿßÁ
 
_user_specified_nameinputs
Ý
B
&__inference_PoolA_layer_call_fn_183779

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolA_layer_call_and_return_conditional_losses_1809402
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿
 
_user_specified_nameinputs
ó
ù
@__inference_ConA_layer_call_and_return_conditional_losses_183730

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿßÁ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿßÁ
 
_user_specified_nameinputs
Ý
B
&__inference_PoolC_layer_call_fn_183933

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolC_layer_call_and_return_conditional_losses_1809772
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ×»:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»
 
_user_specified_nameinputs
¡
]
A__inference_PoolD_layer_call_and_return_conditional_losses_184165

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
d
E__inference_dropout_9_layer_call_and_return_conditional_losses_184325

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ":W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
 
_user_specified_nameinputs
Õ
F
*__inference_flatten_6_layer_call_fn_184466

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿî* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_1812262
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿî2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG>:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>
 
_user_specified_nameinputs
Û
C
'__inference_PoolB2_layer_call_fn_183973

inputs
identityË
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolB2_layer_call_and_return_conditional_losses_1810242
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿF=:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=
 
_user_specified_nameinputs
í
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_183990

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG>:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>
 
_user_specified_nameinputs
¡
]
A__inference_PoolE_layer_call_and_return_conditional_losses_184185

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À
a
(__inference_dropout_layer_call_fn_183826

inputs
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1821292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿI?22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?
 
_user_specified_nameinputs
ó
ù
@__inference_ConB_layer_call_and_return_conditional_losses_183750

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿßÁ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿßÁ
 
_user_specified_nameinputs


%__inference_ConB_layer_call_fn_183759

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ConB_layer_call_and_return_conditional_losses_1809072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿßÁ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿßÁ
 
_user_specified_nameinputs
¡
]
A__inference_PoolC_layer_call_and_return_conditional_losses_180722

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

c
E__inference_dropout_1_layer_call_and_return_conditional_losses_183831

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿH?:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?
 
_user_specified_nameinputs
ó
ù
@__inference_ConD_layer_call_and_return_conditional_losses_184071

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿßÁ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿßÁ
 
_user_specified_nameinputs
á
F
*__inference_dropout_2_layer_call_fn_183995

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_1810182
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG>:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>
 
_user_specified_nameinputs
¡
]
A__inference_PoolE_layer_call_and_return_conditional_losses_180876

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
F
*__inference_dropout_7_layer_call_fn_184276

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_1811802
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
í
d
E__inference_dropout_4_layer_call_and_return_conditional_losses_181973

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿD;:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;
 
_user_specified_nameinputs
Ó
F
*__inference_flatten_5_layer_call_fn_184455

inputs
identityÇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_1812342
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
ö

%__inference_HidC_layer_call_fn_184639

inputs
unknown:î
	unknown_0:	
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_HidC_layer_call_and_return_conditional_losses_1813892
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿî: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿî
 
_user_specified_nameinputs
õ

&__inference_HidB3_layer_call_fn_184619

inputs
unknown:
F
	unknown_0:	
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidB3_layer_call_and_return_conditional_losses_1813722
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿF: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
Ñ
D
(__inference_flatten_layer_call_fn_184400

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ×* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1812742
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ×2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿI?:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?
 
_user_specified_nameinputs

c
E__inference_dropout_7_layer_call_and_return_conditional_losses_184259

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs

õ
@__inference_HidA_layer_call_and_return_conditional_losses_181287

inputs3
matmul_readvariableop_resource:Ê×.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:Ê×*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿÊ×: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ×
 
_user_specified_nameinputs

a
C__inference_dropout_layer_call_and_return_conditional_losses_180971

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿI?:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?
 
_user_specified_nameinputs
í
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_182081

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG>:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>
 
_user_specified_nameinputs


%__inference_ConE_layer_call_fn_184100

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ConE_layer_call_and_return_conditional_losses_1810432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿßÁ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿßÁ
 
_user_specified_nameinputs
è
ú
A__inference_ConB3_layer_call_and_return_conditional_losses_184031

inputs8
conv2d_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
í
d
E__inference_dropout_8_layer_call_and_return_conditional_losses_181904

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
è
ú
A__inference_ConB2_layer_call_and_return_conditional_losses_183904

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿH?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?
 
_user_specified_nameinputs
é
a
E__inference_flatten_9_layer_call_and_return_conditional_losses_181202

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ x 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ð2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ð2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿD;:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;
 
_user_specified_nameinputs
ö

%__inference_HidA_layer_call_fn_184519

inputs
unknown:Ê×
	unknown_0:	
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_HidA_layer_call_and_return_conditional_losses_1812872
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿÊ×: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ×
 
_user_specified_nameinputs

õ
A__inference_HidB3_layer_call_and_return_conditional_losses_184610

inputs2
matmul_readvariableop_resource:
F.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
F*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿF: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
è
ú
A__inference_ConB2_layer_call_and_return_conditional_losses_180990

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿH?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?
 
_user_specified_nameinputs
ø

&__inference_HidE1_layer_call_fn_184699

inputs
unknown: ð
	unknown_0:	
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidE1_layer_call_and_return_conditional_losses_1814402
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ð: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ð
 
_user_specified_nameinputs

c
E__inference_dropout_2_layer_call_and_return_conditional_losses_183978

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG>:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>
 
_user_specified_nameinputs
µ
^
B__inference_PoolA2_layer_call_and_return_conditional_losses_183943

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG=:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=
 
_user_specified_nameinputs


&__inference_ConC2_layer_call_fn_184060

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_ConC2_layer_call_and_return_conditional_losses_1810772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿG>: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>
 
_user_specified_nameinputs
ó
ù
@__inference_ConA_layer_call_and_return_conditional_losses_180924

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿßÁ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿßÁ
 
_user_specified_nameinputs
èï
ÔY
"__inference__traced_restore_185703
file_prefix6
assignvariableop_cona_kernel:*
assignvariableop_1_cona_bias:8
assignvariableop_2_conb_kernel:*
assignvariableop_3_conb_bias:8
assignvariableop_4_conc_kernel:	*
assignvariableop_5_conc_bias:9
assignvariableop_6_cona2_kernel:+
assignvariableop_7_cona2_bias:9
assignvariableop_8_conb2_kernel:+
assignvariableop_9_conb2_bias::
 assignvariableop_10_cona3_kernel:(,
assignvariableop_11_cona3_bias:(:
 assignvariableop_12_conb3_kernel:(,
assignvariableop_13_conb3_bias:(:
 assignvariableop_14_conc2_kernel:,
assignvariableop_15_conc2_bias:9
assignvariableop_16_cond_kernel:	+
assignvariableop_17_cond_bias:9
assignvariableop_18_cone_kernel:+
assignvariableop_19_cone_bias:4
assignvariableop_20_hida_kernel:Ê×,
assignvariableop_21_hida_bias:	5
 assignvariableop_22_hida2_kernel:¤-
assignvariableop_23_hida2_bias:	4
 assignvariableop_24_hida3_kernel:
F-
assignvariableop_25_hida3_bias:	4
assignvariableop_26_hidb_kernel: ©,
assignvariableop_27_hidb_bias:	5
 assignvariableop_28_hidb2_kernel:¤-
assignvariableop_29_hidb2_bias:	4
 assignvariableop_30_hidb3_kernel:
F-
assignvariableop_31_hidb3_bias:	4
assignvariableop_32_hidc_kernel:î,
assignvariableop_33_hidc_bias:	5
 assignvariableop_34_hidc2_kernel:°-
assignvariableop_35_hidc2_bias:	5
 assignvariableop_36_hidd1_kernel:á-
assignvariableop_37_hidd1_bias:	5
 assignvariableop_38_hide1_kernel: ð-
assignvariableop_39_hide1_bias:	3
assignvariableop_40_hid1_kernel:
,
assignvariableop_41_hid1_bias:	3
assignvariableop_42_hid2_kernel:
,
assignvariableop_43_hid2_bias:	4
!assignvariableop_44_output_kernel:	(-
assignvariableop_45_output_bias:('
assignvariableop_46_adam_iter:	 )
assignvariableop_47_adam_beta_1: )
assignvariableop_48_adam_beta_2: (
assignvariableop_49_adam_decay: 0
&assignvariableop_50_adam_learning_rate: #
assignvariableop_51_total: #
assignvariableop_52_count: %
assignvariableop_53_total_1: %
assignvariableop_54_count_1: @
&assignvariableop_55_adam_cona_kernel_m:2
$assignvariableop_56_adam_cona_bias_m:@
&assignvariableop_57_adam_conb_kernel_m:2
$assignvariableop_58_adam_conb_bias_m:@
&assignvariableop_59_adam_conc_kernel_m:	2
$assignvariableop_60_adam_conc_bias_m:A
'assignvariableop_61_adam_cona2_kernel_m:3
%assignvariableop_62_adam_cona2_bias_m:A
'assignvariableop_63_adam_conb2_kernel_m:3
%assignvariableop_64_adam_conb2_bias_m:A
'assignvariableop_65_adam_cona3_kernel_m:(3
%assignvariableop_66_adam_cona3_bias_m:(A
'assignvariableop_67_adam_conb3_kernel_m:(3
%assignvariableop_68_adam_conb3_bias_m:(A
'assignvariableop_69_adam_conc2_kernel_m:3
%assignvariableop_70_adam_conc2_bias_m:@
&assignvariableop_71_adam_cond_kernel_m:	2
$assignvariableop_72_adam_cond_bias_m:@
&assignvariableop_73_adam_cone_kernel_m:2
$assignvariableop_74_adam_cone_bias_m:;
&assignvariableop_75_adam_hida_kernel_m:Ê×3
$assignvariableop_76_adam_hida_bias_m:	<
'assignvariableop_77_adam_hida2_kernel_m:¤4
%assignvariableop_78_adam_hida2_bias_m:	;
'assignvariableop_79_adam_hida3_kernel_m:
F4
%assignvariableop_80_adam_hida3_bias_m:	;
&assignvariableop_81_adam_hidb_kernel_m: ©3
$assignvariableop_82_adam_hidb_bias_m:	<
'assignvariableop_83_adam_hidb2_kernel_m:¤4
%assignvariableop_84_adam_hidb2_bias_m:	;
'assignvariableop_85_adam_hidb3_kernel_m:
F4
%assignvariableop_86_adam_hidb3_bias_m:	;
&assignvariableop_87_adam_hidc_kernel_m:î3
$assignvariableop_88_adam_hidc_bias_m:	<
'assignvariableop_89_adam_hidc2_kernel_m:°4
%assignvariableop_90_adam_hidc2_bias_m:	<
'assignvariableop_91_adam_hidd1_kernel_m:á4
%assignvariableop_92_adam_hidd1_bias_m:	<
'assignvariableop_93_adam_hide1_kernel_m: ð4
%assignvariableop_94_adam_hide1_bias_m:	:
&assignvariableop_95_adam_hid1_kernel_m:
3
$assignvariableop_96_adam_hid1_bias_m:	:
&assignvariableop_97_adam_hid2_kernel_m:
3
$assignvariableop_98_adam_hid2_bias_m:	;
(assignvariableop_99_adam_output_kernel_m:	(5
'assignvariableop_100_adam_output_bias_m:(A
'assignvariableop_101_adam_cona_kernel_v:3
%assignvariableop_102_adam_cona_bias_v:A
'assignvariableop_103_adam_conb_kernel_v:3
%assignvariableop_104_adam_conb_bias_v:A
'assignvariableop_105_adam_conc_kernel_v:	3
%assignvariableop_106_adam_conc_bias_v:B
(assignvariableop_107_adam_cona2_kernel_v:4
&assignvariableop_108_adam_cona2_bias_v:B
(assignvariableop_109_adam_conb2_kernel_v:4
&assignvariableop_110_adam_conb2_bias_v:B
(assignvariableop_111_adam_cona3_kernel_v:(4
&assignvariableop_112_adam_cona3_bias_v:(B
(assignvariableop_113_adam_conb3_kernel_v:(4
&assignvariableop_114_adam_conb3_bias_v:(B
(assignvariableop_115_adam_conc2_kernel_v:4
&assignvariableop_116_adam_conc2_bias_v:A
'assignvariableop_117_adam_cond_kernel_v:	3
%assignvariableop_118_adam_cond_bias_v:A
'assignvariableop_119_adam_cone_kernel_v:3
%assignvariableop_120_adam_cone_bias_v:<
'assignvariableop_121_adam_hida_kernel_v:Ê×4
%assignvariableop_122_adam_hida_bias_v:	=
(assignvariableop_123_adam_hida2_kernel_v:¤5
&assignvariableop_124_adam_hida2_bias_v:	<
(assignvariableop_125_adam_hida3_kernel_v:
F5
&assignvariableop_126_adam_hida3_bias_v:	<
'assignvariableop_127_adam_hidb_kernel_v: ©4
%assignvariableop_128_adam_hidb_bias_v:	=
(assignvariableop_129_adam_hidb2_kernel_v:¤5
&assignvariableop_130_adam_hidb2_bias_v:	<
(assignvariableop_131_adam_hidb3_kernel_v:
F5
&assignvariableop_132_adam_hidb3_bias_v:	<
'assignvariableop_133_adam_hidc_kernel_v:î4
%assignvariableop_134_adam_hidc_bias_v:	=
(assignvariableop_135_adam_hidc2_kernel_v:°5
&assignvariableop_136_adam_hidc2_bias_v:	=
(assignvariableop_137_adam_hidd1_kernel_v:á5
&assignvariableop_138_adam_hidd1_bias_v:	=
(assignvariableop_139_adam_hide1_kernel_v: ð5
&assignvariableop_140_adam_hide1_bias_v:	;
'assignvariableop_141_adam_hid1_kernel_v:
4
%assignvariableop_142_adam_hid1_bias_v:	;
'assignvariableop_143_adam_hid2_kernel_v:
4
%assignvariableop_144_adam_hid2_bias_v:	<
)assignvariableop_145_adam_output_kernel_v:	(5
'assignvariableop_146_adam_output_bias_v:(
identity_148¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_100¢AssignVariableOp_101¢AssignVariableOp_102¢AssignVariableOp_103¢AssignVariableOp_104¢AssignVariableOp_105¢AssignVariableOp_106¢AssignVariableOp_107¢AssignVariableOp_108¢AssignVariableOp_109¢AssignVariableOp_11¢AssignVariableOp_110¢AssignVariableOp_111¢AssignVariableOp_112¢AssignVariableOp_113¢AssignVariableOp_114¢AssignVariableOp_115¢AssignVariableOp_116¢AssignVariableOp_117¢AssignVariableOp_118¢AssignVariableOp_119¢AssignVariableOp_12¢AssignVariableOp_120¢AssignVariableOp_121¢AssignVariableOp_122¢AssignVariableOp_123¢AssignVariableOp_124¢AssignVariableOp_125¢AssignVariableOp_126¢AssignVariableOp_127¢AssignVariableOp_128¢AssignVariableOp_129¢AssignVariableOp_13¢AssignVariableOp_130¢AssignVariableOp_131¢AssignVariableOp_132¢AssignVariableOp_133¢AssignVariableOp_134¢AssignVariableOp_135¢AssignVariableOp_136¢AssignVariableOp_137¢AssignVariableOp_138¢AssignVariableOp_139¢AssignVariableOp_14¢AssignVariableOp_140¢AssignVariableOp_141¢AssignVariableOp_142¢AssignVariableOp_143¢AssignVariableOp_144¢AssignVariableOp_145¢AssignVariableOp_146¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_87¢AssignVariableOp_88¢AssignVariableOp_89¢AssignVariableOp_9¢AssignVariableOp_90¢AssignVariableOp_91¢AssignVariableOp_92¢AssignVariableOp_93¢AssignVariableOp_94¢AssignVariableOp_95¢AssignVariableOp_96¢AssignVariableOp_97¢AssignVariableOp_98¢AssignVariableOp_99êT
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*õS
valueëSBèSB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names»
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*¾
value´B±B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*æ
_output_shapesÓ
Ð::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*¥
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_cona_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¡
AssignVariableOp_1AssignVariableOpassignvariableop_1_cona_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2£
AssignVariableOp_2AssignVariableOpassignvariableop_2_conb_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¡
AssignVariableOp_3AssignVariableOpassignvariableop_3_conb_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4£
AssignVariableOp_4AssignVariableOpassignvariableop_4_conc_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¡
AssignVariableOp_5AssignVariableOpassignvariableop_5_conc_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¤
AssignVariableOp_6AssignVariableOpassignvariableop_6_cona2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¢
AssignVariableOp_7AssignVariableOpassignvariableop_7_cona2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¤
AssignVariableOp_8AssignVariableOpassignvariableop_8_conb2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¢
AssignVariableOp_9AssignVariableOpassignvariableop_9_conb2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¨
AssignVariableOp_10AssignVariableOp assignvariableop_10_cona3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¦
AssignVariableOp_11AssignVariableOpassignvariableop_11_cona3_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¨
AssignVariableOp_12AssignVariableOp assignvariableop_12_conb3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¦
AssignVariableOp_13AssignVariableOpassignvariableop_13_conb3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¨
AssignVariableOp_14AssignVariableOp assignvariableop_14_conc2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¦
AssignVariableOp_15AssignVariableOpassignvariableop_15_conc2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16§
AssignVariableOp_16AssignVariableOpassignvariableop_16_cond_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¥
AssignVariableOp_17AssignVariableOpassignvariableop_17_cond_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18§
AssignVariableOp_18AssignVariableOpassignvariableop_18_cone_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¥
AssignVariableOp_19AssignVariableOpassignvariableop_19_cone_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20§
AssignVariableOp_20AssignVariableOpassignvariableop_20_hida_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¥
AssignVariableOp_21AssignVariableOpassignvariableop_21_hida_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¨
AssignVariableOp_22AssignVariableOp assignvariableop_22_hida2_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¦
AssignVariableOp_23AssignVariableOpassignvariableop_23_hida2_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¨
AssignVariableOp_24AssignVariableOp assignvariableop_24_hida3_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¦
AssignVariableOp_25AssignVariableOpassignvariableop_25_hida3_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26§
AssignVariableOp_26AssignVariableOpassignvariableop_26_hidb_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¥
AssignVariableOp_27AssignVariableOpassignvariableop_27_hidb_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28¨
AssignVariableOp_28AssignVariableOp assignvariableop_28_hidb2_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29¦
AssignVariableOp_29AssignVariableOpassignvariableop_29_hidb2_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¨
AssignVariableOp_30AssignVariableOp assignvariableop_30_hidb3_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31¦
AssignVariableOp_31AssignVariableOpassignvariableop_31_hidb3_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32§
AssignVariableOp_32AssignVariableOpassignvariableop_32_hidc_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33¥
AssignVariableOp_33AssignVariableOpassignvariableop_33_hidc_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34¨
AssignVariableOp_34AssignVariableOp assignvariableop_34_hidc2_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35¦
AssignVariableOp_35AssignVariableOpassignvariableop_35_hidc2_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36¨
AssignVariableOp_36AssignVariableOp assignvariableop_36_hidd1_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37¦
AssignVariableOp_37AssignVariableOpassignvariableop_37_hidd1_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38¨
AssignVariableOp_38AssignVariableOp assignvariableop_38_hide1_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39¦
AssignVariableOp_39AssignVariableOpassignvariableop_39_hide1_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40§
AssignVariableOp_40AssignVariableOpassignvariableop_40_hid1_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41¥
AssignVariableOp_41AssignVariableOpassignvariableop_41_hid1_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42§
AssignVariableOp_42AssignVariableOpassignvariableop_42_hid2_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43¥
AssignVariableOp_43AssignVariableOpassignvariableop_43_hid2_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44©
AssignVariableOp_44AssignVariableOp!assignvariableop_44_output_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45§
AssignVariableOp_45AssignVariableOpassignvariableop_45_output_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_46¥
AssignVariableOp_46AssignVariableOpassignvariableop_46_adam_iterIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47§
AssignVariableOp_47AssignVariableOpassignvariableop_47_adam_beta_1Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48§
AssignVariableOp_48AssignVariableOpassignvariableop_48_adam_beta_2Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49¦
AssignVariableOp_49AssignVariableOpassignvariableop_49_adam_decayIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50®
AssignVariableOp_50AssignVariableOp&assignvariableop_50_adam_learning_rateIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51¡
AssignVariableOp_51AssignVariableOpassignvariableop_51_totalIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52¡
AssignVariableOp_52AssignVariableOpassignvariableop_52_countIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53£
AssignVariableOp_53AssignVariableOpassignvariableop_53_total_1Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54£
AssignVariableOp_54AssignVariableOpassignvariableop_54_count_1Identity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55®
AssignVariableOp_55AssignVariableOp&assignvariableop_55_adam_cona_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56¬
AssignVariableOp_56AssignVariableOp$assignvariableop_56_adam_cona_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57®
AssignVariableOp_57AssignVariableOp&assignvariableop_57_adam_conb_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58¬
AssignVariableOp_58AssignVariableOp$assignvariableop_58_adam_conb_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59®
AssignVariableOp_59AssignVariableOp&assignvariableop_59_adam_conc_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60¬
AssignVariableOp_60AssignVariableOp$assignvariableop_60_adam_conc_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61¯
AssignVariableOp_61AssignVariableOp'assignvariableop_61_adam_cona2_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62­
AssignVariableOp_62AssignVariableOp%assignvariableop_62_adam_cona2_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63¯
AssignVariableOp_63AssignVariableOp'assignvariableop_63_adam_conb2_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64­
AssignVariableOp_64AssignVariableOp%assignvariableop_64_adam_conb2_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65¯
AssignVariableOp_65AssignVariableOp'assignvariableop_65_adam_cona3_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66­
AssignVariableOp_66AssignVariableOp%assignvariableop_66_adam_cona3_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67¯
AssignVariableOp_67AssignVariableOp'assignvariableop_67_adam_conb3_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68­
AssignVariableOp_68AssignVariableOp%assignvariableop_68_adam_conb3_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69¯
AssignVariableOp_69AssignVariableOp'assignvariableop_69_adam_conc2_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70­
AssignVariableOp_70AssignVariableOp%assignvariableop_70_adam_conc2_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71®
AssignVariableOp_71AssignVariableOp&assignvariableop_71_adam_cond_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72¬
AssignVariableOp_72AssignVariableOp$assignvariableop_72_adam_cond_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73®
AssignVariableOp_73AssignVariableOp&assignvariableop_73_adam_cone_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74¬
AssignVariableOp_74AssignVariableOp$assignvariableop_74_adam_cone_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75®
AssignVariableOp_75AssignVariableOp&assignvariableop_75_adam_hida_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76¬
AssignVariableOp_76AssignVariableOp$assignvariableop_76_adam_hida_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77¯
AssignVariableOp_77AssignVariableOp'assignvariableop_77_adam_hida2_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78­
AssignVariableOp_78AssignVariableOp%assignvariableop_78_adam_hida2_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79¯
AssignVariableOp_79AssignVariableOp'assignvariableop_79_adam_hida3_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80­
AssignVariableOp_80AssignVariableOp%assignvariableop_80_adam_hida3_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81®
AssignVariableOp_81AssignVariableOp&assignvariableop_81_adam_hidb_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82¬
AssignVariableOp_82AssignVariableOp$assignvariableop_82_adam_hidb_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83¯
AssignVariableOp_83AssignVariableOp'assignvariableop_83_adam_hidb2_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84­
AssignVariableOp_84AssignVariableOp%assignvariableop_84_adam_hidb2_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85¯
AssignVariableOp_85AssignVariableOp'assignvariableop_85_adam_hidb3_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86­
AssignVariableOp_86AssignVariableOp%assignvariableop_86_adam_hidb3_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87®
AssignVariableOp_87AssignVariableOp&assignvariableop_87_adam_hidc_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88¬
AssignVariableOp_88AssignVariableOp$assignvariableop_88_adam_hidc_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89¯
AssignVariableOp_89AssignVariableOp'assignvariableop_89_adam_hidc2_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90­
AssignVariableOp_90AssignVariableOp%assignvariableop_90_adam_hidc2_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91¯
AssignVariableOp_91AssignVariableOp'assignvariableop_91_adam_hidd1_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92­
AssignVariableOp_92AssignVariableOp%assignvariableop_92_adam_hidd1_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93¯
AssignVariableOp_93AssignVariableOp'assignvariableop_93_adam_hide1_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94­
AssignVariableOp_94AssignVariableOp%assignvariableop_94_adam_hide1_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95®
AssignVariableOp_95AssignVariableOp&assignvariableop_95_adam_hid1_kernel_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96¬
AssignVariableOp_96AssignVariableOp$assignvariableop_96_adam_hid1_bias_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97®
AssignVariableOp_97AssignVariableOp&assignvariableop_97_adam_hid2_kernel_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98¬
AssignVariableOp_98AssignVariableOp$assignvariableop_98_adam_hid2_bias_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99°
AssignVariableOp_99AssignVariableOp(assignvariableop_99_adam_output_kernel_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100²
AssignVariableOp_100AssignVariableOp'assignvariableop_100_adam_output_bias_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101²
AssignVariableOp_101AssignVariableOp'assignvariableop_101_adam_cona_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102°
AssignVariableOp_102AssignVariableOp%assignvariableop_102_adam_cona_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103²
AssignVariableOp_103AssignVariableOp'assignvariableop_103_adam_conb_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104°
AssignVariableOp_104AssignVariableOp%assignvariableop_104_adam_conb_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105²
AssignVariableOp_105AssignVariableOp'assignvariableop_105_adam_conc_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106°
AssignVariableOp_106AssignVariableOp%assignvariableop_106_adam_conc_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107³
AssignVariableOp_107AssignVariableOp(assignvariableop_107_adam_cona2_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108±
AssignVariableOp_108AssignVariableOp&assignvariableop_108_adam_cona2_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109³
AssignVariableOp_109AssignVariableOp(assignvariableop_109_adam_conb2_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110±
AssignVariableOp_110AssignVariableOp&assignvariableop_110_adam_conb2_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111³
AssignVariableOp_111AssignVariableOp(assignvariableop_111_adam_cona3_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112±
AssignVariableOp_112AssignVariableOp&assignvariableop_112_adam_cona3_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113³
AssignVariableOp_113AssignVariableOp(assignvariableop_113_adam_conb3_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114±
AssignVariableOp_114AssignVariableOp&assignvariableop_114_adam_conb3_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115³
AssignVariableOp_115AssignVariableOp(assignvariableop_115_adam_conc2_kernel_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116±
AssignVariableOp_116AssignVariableOp&assignvariableop_116_adam_conc2_bias_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117²
AssignVariableOp_117AssignVariableOp'assignvariableop_117_adam_cond_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118°
AssignVariableOp_118AssignVariableOp%assignvariableop_118_adam_cond_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119²
AssignVariableOp_119AssignVariableOp'assignvariableop_119_adam_cone_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120°
AssignVariableOp_120AssignVariableOp%assignvariableop_120_adam_cone_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121²
AssignVariableOp_121AssignVariableOp'assignvariableop_121_adam_hida_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122°
AssignVariableOp_122AssignVariableOp%assignvariableop_122_adam_hida_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123³
AssignVariableOp_123AssignVariableOp(assignvariableop_123_adam_hida2_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124±
AssignVariableOp_124AssignVariableOp&assignvariableop_124_adam_hida2_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125³
AssignVariableOp_125AssignVariableOp(assignvariableop_125_adam_hida3_kernel_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126±
AssignVariableOp_126AssignVariableOp&assignvariableop_126_adam_hida3_bias_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127²
AssignVariableOp_127AssignVariableOp'assignvariableop_127_adam_hidb_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128°
AssignVariableOp_128AssignVariableOp%assignvariableop_128_adam_hidb_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129³
AssignVariableOp_129AssignVariableOp(assignvariableop_129_adam_hidb2_kernel_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130±
AssignVariableOp_130AssignVariableOp&assignvariableop_130_adam_hidb2_bias_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131³
AssignVariableOp_131AssignVariableOp(assignvariableop_131_adam_hidb3_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132±
AssignVariableOp_132AssignVariableOp&assignvariableop_132_adam_hidb3_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_132q
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:2
Identity_133²
AssignVariableOp_133AssignVariableOp'assignvariableop_133_adam_hidc_kernel_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_133q
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:2
Identity_134°
AssignVariableOp_134AssignVariableOp%assignvariableop_134_adam_hidc_bias_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_134q
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:2
Identity_135³
AssignVariableOp_135AssignVariableOp(assignvariableop_135_adam_hidc2_kernel_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_135q
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:2
Identity_136±
AssignVariableOp_136AssignVariableOp&assignvariableop_136_adam_hidc2_bias_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_136q
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:2
Identity_137³
AssignVariableOp_137AssignVariableOp(assignvariableop_137_adam_hidd1_kernel_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_137q
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:2
Identity_138±
AssignVariableOp_138AssignVariableOp&assignvariableop_138_adam_hidd1_bias_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_138q
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:2
Identity_139³
AssignVariableOp_139AssignVariableOp(assignvariableop_139_adam_hide1_kernel_vIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139q
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:2
Identity_140±
AssignVariableOp_140AssignVariableOp&assignvariableop_140_adam_hide1_bias_vIdentity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_140q
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:2
Identity_141²
AssignVariableOp_141AssignVariableOp'assignvariableop_141_adam_hid1_kernel_vIdentity_141:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_141q
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:2
Identity_142°
AssignVariableOp_142AssignVariableOp%assignvariableop_142_adam_hid1_bias_vIdentity_142:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_142q
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:2
Identity_143²
AssignVariableOp_143AssignVariableOp'assignvariableop_143_adam_hid2_kernel_vIdentity_143:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_143q
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:2
Identity_144°
AssignVariableOp_144AssignVariableOp%assignvariableop_144_adam_hid2_bias_vIdentity_144:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_144q
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
:2
Identity_145´
AssignVariableOp_145AssignVariableOp)assignvariableop_145_adam_output_kernel_vIdentity_145:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_145q
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:2
Identity_146²
AssignVariableOp_146AssignVariableOp'assignvariableop_146_adam_output_bias_vIdentity_146:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1469
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp±
Identity_147Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_147i
Identity_148IdentityIdentity_147:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_148
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"%
identity_148Identity_148:output:0*½
_input_shapes«
¨: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_146AssignVariableOp_1462*
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
í
d
E__inference_dropout_4_layer_call_and_return_conditional_losses_184379

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿD;:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;
 
_user_specified_nameinputs

c
E__inference_dropout_4_layer_call_and_return_conditional_losses_184367

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿD;:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;
 
_user_specified_nameinputs
á
F
*__inference_dropout_8_layer_call_fn_184303

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_1811732
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
Ë
î7
__inference__traced_save_185252
file_prefix*
&savev2_cona_kernel_read_readvariableop(
$savev2_cona_bias_read_readvariableop*
&savev2_conb_kernel_read_readvariableop(
$savev2_conb_bias_read_readvariableop*
&savev2_conc_kernel_read_readvariableop(
$savev2_conc_bias_read_readvariableop+
'savev2_cona2_kernel_read_readvariableop)
%savev2_cona2_bias_read_readvariableop+
'savev2_conb2_kernel_read_readvariableop)
%savev2_conb2_bias_read_readvariableop+
'savev2_cona3_kernel_read_readvariableop)
%savev2_cona3_bias_read_readvariableop+
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
%savev2_hida3_bias_read_readvariableop*
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
+savev2_adam_conb_bias_m_read_readvariableop1
-savev2_adam_conc_kernel_m_read_readvariableop/
+savev2_adam_conc_bias_m_read_readvariableop2
.savev2_adam_cona2_kernel_m_read_readvariableop0
,savev2_adam_cona2_bias_m_read_readvariableop2
.savev2_adam_conb2_kernel_m_read_readvariableop0
,savev2_adam_conb2_bias_m_read_readvariableop2
.savev2_adam_cona3_kernel_m_read_readvariableop0
,savev2_adam_cona3_bias_m_read_readvariableop2
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
,savev2_adam_hida3_bias_m_read_readvariableop1
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
+savev2_adam_conb_bias_v_read_readvariableop1
-savev2_adam_conc_kernel_v_read_readvariableop/
+savev2_adam_conc_bias_v_read_readvariableop2
.savev2_adam_cona2_kernel_v_read_readvariableop0
,savev2_adam_cona2_bias_v_read_readvariableop2
.savev2_adam_conb2_kernel_v_read_readvariableop0
,savev2_adam_conb2_bias_v_read_readvariableop2
.savev2_adam_cona3_kernel_v_read_readvariableop0
,savev2_adam_cona3_bias_v_read_readvariableop2
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
,savev2_adam_hida3_bias_v_read_readvariableop1
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

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameäT
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*õS
valueëSBèSB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesµ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*¾
value´B±B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices5
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_cona_kernel_read_readvariableop$savev2_cona_bias_read_readvariableop&savev2_conb_kernel_read_readvariableop$savev2_conb_bias_read_readvariableop&savev2_conc_kernel_read_readvariableop$savev2_conc_bias_read_readvariableop'savev2_cona2_kernel_read_readvariableop%savev2_cona2_bias_read_readvariableop'savev2_conb2_kernel_read_readvariableop%savev2_conb2_bias_read_readvariableop'savev2_cona3_kernel_read_readvariableop%savev2_cona3_bias_read_readvariableop'savev2_conb3_kernel_read_readvariableop%savev2_conb3_bias_read_readvariableop'savev2_conc2_kernel_read_readvariableop%savev2_conc2_bias_read_readvariableop&savev2_cond_kernel_read_readvariableop$savev2_cond_bias_read_readvariableop&savev2_cone_kernel_read_readvariableop$savev2_cone_bias_read_readvariableop&savev2_hida_kernel_read_readvariableop$savev2_hida_bias_read_readvariableop'savev2_hida2_kernel_read_readvariableop%savev2_hida2_bias_read_readvariableop'savev2_hida3_kernel_read_readvariableop%savev2_hida3_bias_read_readvariableop&savev2_hidb_kernel_read_readvariableop$savev2_hidb_bias_read_readvariableop'savev2_hidb2_kernel_read_readvariableop%savev2_hidb2_bias_read_readvariableop'savev2_hidb3_kernel_read_readvariableop%savev2_hidb3_bias_read_readvariableop&savev2_hidc_kernel_read_readvariableop$savev2_hidc_bias_read_readvariableop'savev2_hidc2_kernel_read_readvariableop%savev2_hidc2_bias_read_readvariableop'savev2_hidd1_kernel_read_readvariableop%savev2_hidd1_bias_read_readvariableop'savev2_hide1_kernel_read_readvariableop%savev2_hide1_bias_read_readvariableop&savev2_hid1_kernel_read_readvariableop$savev2_hid1_bias_read_readvariableop&savev2_hid2_kernel_read_readvariableop$savev2_hid2_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop-savev2_adam_cona_kernel_m_read_readvariableop+savev2_adam_cona_bias_m_read_readvariableop-savev2_adam_conb_kernel_m_read_readvariableop+savev2_adam_conb_bias_m_read_readvariableop-savev2_adam_conc_kernel_m_read_readvariableop+savev2_adam_conc_bias_m_read_readvariableop.savev2_adam_cona2_kernel_m_read_readvariableop,savev2_adam_cona2_bias_m_read_readvariableop.savev2_adam_conb2_kernel_m_read_readvariableop,savev2_adam_conb2_bias_m_read_readvariableop.savev2_adam_cona3_kernel_m_read_readvariableop,savev2_adam_cona3_bias_m_read_readvariableop.savev2_adam_conb3_kernel_m_read_readvariableop,savev2_adam_conb3_bias_m_read_readvariableop.savev2_adam_conc2_kernel_m_read_readvariableop,savev2_adam_conc2_bias_m_read_readvariableop-savev2_adam_cond_kernel_m_read_readvariableop+savev2_adam_cond_bias_m_read_readvariableop-savev2_adam_cone_kernel_m_read_readvariableop+savev2_adam_cone_bias_m_read_readvariableop-savev2_adam_hida_kernel_m_read_readvariableop+savev2_adam_hida_bias_m_read_readvariableop.savev2_adam_hida2_kernel_m_read_readvariableop,savev2_adam_hida2_bias_m_read_readvariableop.savev2_adam_hida3_kernel_m_read_readvariableop,savev2_adam_hida3_bias_m_read_readvariableop-savev2_adam_hidb_kernel_m_read_readvariableop+savev2_adam_hidb_bias_m_read_readvariableop.savev2_adam_hidb2_kernel_m_read_readvariableop,savev2_adam_hidb2_bias_m_read_readvariableop.savev2_adam_hidb3_kernel_m_read_readvariableop,savev2_adam_hidb3_bias_m_read_readvariableop-savev2_adam_hidc_kernel_m_read_readvariableop+savev2_adam_hidc_bias_m_read_readvariableop.savev2_adam_hidc2_kernel_m_read_readvariableop,savev2_adam_hidc2_bias_m_read_readvariableop.savev2_adam_hidd1_kernel_m_read_readvariableop,savev2_adam_hidd1_bias_m_read_readvariableop.savev2_adam_hide1_kernel_m_read_readvariableop,savev2_adam_hide1_bias_m_read_readvariableop-savev2_adam_hid1_kernel_m_read_readvariableop+savev2_adam_hid1_bias_m_read_readvariableop-savev2_adam_hid2_kernel_m_read_readvariableop+savev2_adam_hid2_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop-savev2_adam_cona_kernel_v_read_readvariableop+savev2_adam_cona_bias_v_read_readvariableop-savev2_adam_conb_kernel_v_read_readvariableop+savev2_adam_conb_bias_v_read_readvariableop-savev2_adam_conc_kernel_v_read_readvariableop+savev2_adam_conc_bias_v_read_readvariableop.savev2_adam_cona2_kernel_v_read_readvariableop,savev2_adam_cona2_bias_v_read_readvariableop.savev2_adam_conb2_kernel_v_read_readvariableop,savev2_adam_conb2_bias_v_read_readvariableop.savev2_adam_cona3_kernel_v_read_readvariableop,savev2_adam_cona3_bias_v_read_readvariableop.savev2_adam_conb3_kernel_v_read_readvariableop,savev2_adam_conb3_bias_v_read_readvariableop.savev2_adam_conc2_kernel_v_read_readvariableop,savev2_adam_conc2_bias_v_read_readvariableop-savev2_adam_cond_kernel_v_read_readvariableop+savev2_adam_cond_bias_v_read_readvariableop-savev2_adam_cone_kernel_v_read_readvariableop+savev2_adam_cone_bias_v_read_readvariableop-savev2_adam_hida_kernel_v_read_readvariableop+savev2_adam_hida_bias_v_read_readvariableop.savev2_adam_hida2_kernel_v_read_readvariableop,savev2_adam_hida2_bias_v_read_readvariableop.savev2_adam_hida3_kernel_v_read_readvariableop,savev2_adam_hida3_bias_v_read_readvariableop-savev2_adam_hidb_kernel_v_read_readvariableop+savev2_adam_hidb_bias_v_read_readvariableop.savev2_adam_hidb2_kernel_v_read_readvariableop,savev2_adam_hidb2_bias_v_read_readvariableop.savev2_adam_hidb3_kernel_v_read_readvariableop,savev2_adam_hidb3_bias_v_read_readvariableop-savev2_adam_hidc_kernel_v_read_readvariableop+savev2_adam_hidc_bias_v_read_readvariableop.savev2_adam_hidc2_kernel_v_read_readvariableop,savev2_adam_hidc2_bias_v_read_readvariableop.savev2_adam_hidd1_kernel_v_read_readvariableop,savev2_adam_hidd1_bias_v_read_readvariableop.savev2_adam_hide1_kernel_v_read_readvariableop,savev2_adam_hide1_bias_v_read_readvariableop-savev2_adam_hid1_kernel_v_read_readvariableop+savev2_adam_hid1_bias_v_read_readvariableop-savev2_adam_hid2_kernel_v_read_readvariableop+savev2_adam_hid2_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *¥
dtypes
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*ò
_input_shapesà
Ý: :::::	::::::(:(:(:(:::	::::Ê×::¤::
F:: ©::¤::
F::î::°::á:: ð::
::
::	(:(: : : : : : : : : :::::	::::::(:(:(:(:::	::::Ê×::¤::
F:: ©::¤::
F::î::°::á:: ð::
::
::	(:(:::::	::::::(:(:(:(:::	::::Ê×::¤::
F:: ©::¤::
F::î::°::á:: ð::
::
::	(:(: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:	: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
:: 


_output_shapes
::,(
&
_output_shapes
:(: 

_output_shapes
:(:,(
&
_output_shapes
:(: 

_output_shapes
:(:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:	: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::'#
!
_output_shapes
:Ê×:!

_output_shapes	
::'#
!
_output_shapes
:¤:!

_output_shapes	
::&"
 
_output_shapes
:
F:!

_output_shapes	
::'#
!
_output_shapes
: ©:!

_output_shapes	
::'#
!
_output_shapes
:¤:!

_output_shapes	
::&"
 
_output_shapes
:
F:! 

_output_shapes	
::'!#
!
_output_shapes
:î:!"

_output_shapes	
::'##
!
_output_shapes
:°:!$

_output_shapes	
::'%#
!
_output_shapes
:á:!&

_output_shapes	
::''#
!
_output_shapes
: ð:!(

_output_shapes	
::&)"
 
_output_shapes
:
:!*

_output_shapes	
::&+"
 
_output_shapes
:
:!,

_output_shapes	
::%-!

_output_shapes
:	(: .

_output_shapes
:(:/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3
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
: :,8(
&
_output_shapes
:: 9

_output_shapes
::,:(
&
_output_shapes
:: ;

_output_shapes
::,<(
&
_output_shapes
:	: =

_output_shapes
::,>(
&
_output_shapes
:: ?

_output_shapes
::,@(
&
_output_shapes
:: A

_output_shapes
::,B(
&
_output_shapes
:(: C

_output_shapes
:(:,D(
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
:	: I

_output_shapes
::,J(
&
_output_shapes
:: K

_output_shapes
::'L#
!
_output_shapes
:Ê×:!M

_output_shapes	
::'N#
!
_output_shapes
:¤:!O

_output_shapes	
::&P"
 
_output_shapes
:
F:!Q

_output_shapes	
::'R#
!
_output_shapes
: ©:!S

_output_shapes	
::'T#
!
_output_shapes
:¤:!U

_output_shapes	
::&V"
 
_output_shapes
:
F:!W

_output_shapes	
::'X#
!
_output_shapes
:î:!Y

_output_shapes	
::'Z#
!
_output_shapes
:°:![

_output_shapes	
::'\#
!
_output_shapes
:á:!]

_output_shapes	
::'^#
!
_output_shapes
: ð:!_

_output_shapes	
::&`"
 
_output_shapes
:
:!a

_output_shapes	
::&b"
 
_output_shapes
:
:!c

_output_shapes	
::%d!

_output_shapes
:	(: e

_output_shapes
:(:,f(
&
_output_shapes
:: g

_output_shapes
::,h(
&
_output_shapes
:: i

_output_shapes
::,j(
&
_output_shapes
:	: k

_output_shapes
::,l(
&
_output_shapes
:: m

_output_shapes
::,n(
&
_output_shapes
:: o

_output_shapes
::,p(
&
_output_shapes
:(: q

_output_shapes
:(:,r(
&
_output_shapes
:(: s

_output_shapes
:(:,t(
&
_output_shapes
:: u

_output_shapes
::,v(
&
_output_shapes
:	: w

_output_shapes
::,x(
&
_output_shapes
:: y

_output_shapes
::'z#
!
_output_shapes
:Ê×:!{

_output_shapes	
::'|#
!
_output_shapes
:¤:!}

_output_shapes	
::&~"
 
_output_shapes
:
F:!

_output_shapes	
::(#
!
_output_shapes
: ©:"

_output_shapes	
::(#
!
_output_shapes
:¤:"

_output_shapes	
::'"
 
_output_shapes
:
F:"

_output_shapes	
::(#
!
_output_shapes
:î:"

_output_shapes	
::(#
!
_output_shapes
:°:"

_output_shapes	
::(#
!
_output_shapes
:á:"

_output_shapes	
::(#
!
_output_shapes
: ð:"

_output_shapes	
::'"
 
_output_shapes
:
:"

_output_shapes	
::'"
 
_output_shapes
:
:"

_output_shapes	
::&!

_output_shapes
:	(:!

_output_shapes
:(:

_output_shapes
: 
ó
ù
@__inference_ConC_layer_call_and_return_conditional_losses_183864

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿßÁ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿßÁ
 
_user_specified_nameinputs
È
C
'__inference_PoolA3_layer_call_fn_184115

inputs
identityæ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolA3_layer_call_and_return_conditional_losses_1807882
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
]
A__inference_PoolA_layer_call_and_return_conditional_losses_180940

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿
 
_user_specified_nameinputs
ó
ù
@__inference_ConE_layer_call_and_return_conditional_losses_181043

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿßÁ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿßÁ
 
_user_specified_nameinputs
Ý
B
&__inference_PoolE_layer_call_fn_184200

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolE_layer_call_and_return_conditional_losses_1811212
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÍ³:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³
 
_user_specified_nameinputs
é
a
E__inference_flatten_9_layer_call_and_return_conditional_losses_184494

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ x 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ð2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ð2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿD;:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;
 
_user_specified_nameinputs
¡
]
A__inference_PoolC_layer_call_and_return_conditional_losses_183918

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


%__inference_ConC_layer_call_fn_183873

inputs!
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ConC_layer_call_and_return_conditional_losses_1809532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿßÁ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿßÁ
 
_user_specified_nameinputs

õ
@__inference_HidC_layer_call_and_return_conditional_losses_181389

inputs3
matmul_readvariableop_resource:î.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:î*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿî: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿî
 
_user_specified_nameinputs

ö
A__inference_HidA2_layer_call_and_return_conditional_losses_184530

inputs3
matmul_readvariableop_resource:¤.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:¤*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¤: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
 
_user_specified_nameinputs

ô
@__inference_Hid2_layer_call_and_return_conditional_losses_181491

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanhd
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢
^
B__inference_PoolB2_layer_call_and_return_conditional_losses_183958

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢
^
B__inference_PoolB2_layer_call_and_return_conditional_losses_180766

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ
B
&__inference_PoolE_layer_call_fn_184195

inputs
identityå
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolE_layer_call_and_return_conditional_losses_1808762
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë
b
C__inference_dropout_layer_call_and_return_conditional_losses_182129

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿI?:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?
 
_user_specified_nameinputs
Õ
F
*__inference_flatten_8_layer_call_fn_184488

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿá* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_1812102
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿá2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG=:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=
 
_user_specified_nameinputs

c
E__inference_dropout_5_layer_call_and_return_conditional_losses_184205

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs

ö
A__inference_HidD1_layer_call_and_return_conditional_losses_184670

inputs3
matmul_readvariableop_resource:á.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:á*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿá: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿá
 
_user_specified_nameinputs
ø

&__inference_HidD1_layer_call_fn_184679

inputs
unknown:á
	unknown_0:	
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidD1_layer_call_and_return_conditional_losses_1814232
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿá: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿá
 
_user_specified_nameinputs
¸
]
A__inference_PoolD_layer_call_and_return_conditional_losses_184170

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹
 
_user_specified_nameinputs
Ä
c
*__inference_dropout_5_layer_call_fn_184227

inputs
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_1818352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
È
C
'__inference_PoolB2_layer_call_fn_183968

inputs
identityæ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolB2_layer_call_and_return_conditional_losses_1807662
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø

&__inference_HidA2_layer_call_fn_184539

inputs
unknown:¤
	unknown_0:	
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidA2_layer_call_and_return_conditional_losses_1813042
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¤: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
 
_user_specified_nameinputs
í
d
E__inference_dropout_7_layer_call_and_return_conditional_losses_181881

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
é
a
E__inference_flatten_8_layer_call_and_return_conditional_losses_184483

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ0 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿá2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿá2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG=:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=
 
_user_specified_nameinputs
Ä
c
*__inference_dropout_1_layer_call_fn_183853

inputs
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1821522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿH?22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?
 
_user_specified_nameinputs

õ
@__inference_HidB_layer_call_and_return_conditional_losses_184570

inputs3
matmul_readvariableop_resource: ©.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
: ©*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ©: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©
 
_user_specified_nameinputs
Û
C
'__inference_PoolA3_layer_call_fn_184120

inputs
identityË
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolA3_layer_call_and_return_conditional_losses_1811452
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ!(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(
 
_user_specified_nameinputs

c
E__inference_dropout_1_layer_call_and_return_conditional_losses_180964

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿH?:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?
 
_user_specified_nameinputs
¡
]
A__inference_PoolB_layer_call_and_return_conditional_losses_183784

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

c
E__inference_dropout_8_layer_call_and_return_conditional_losses_181173

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs


&__inference_ConA3_layer_call_fn_184020

inputs!
unknown:(
	unknown_0:(
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_ConA3_layer_call_and_return_conditional_losses_1811112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ#: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
ó
ù
@__inference_ConD_layer_call_and_return_conditional_losses_181060

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿßÁ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿßÁ
 
_user_specified_nameinputs
µ
^
B__inference_PoolA2_layer_call_and_return_conditional_losses_181030

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG=:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=
 
_user_specified_nameinputs
¢
^
B__inference_PoolB3_layer_call_and_return_conditional_losses_180810

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

c
E__inference_dropout_5_layer_call_and_return_conditional_losses_181194

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs

ô
B__inference_output_layer_call_and_return_conditional_losses_181508

inputs1
matmul_readvariableop_resource:	(-
biasadd_readvariableop_resource:(
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
]
A__inference_PoolC_layer_call_and_return_conditional_losses_180977

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ×»:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»
 
_user_specified_nameinputs
¢
^
B__inference_PoolC2_layer_call_and_return_conditional_losses_180832

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È
C
'__inference_PoolB3_layer_call_fn_184135

inputs
identityæ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolB3_layer_call_and_return_conditional_losses_1808102
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é
a
E__inference_flatten_4_layer_call_and_return_conditional_losses_184439

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿR  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
Ä
c
*__inference_dropout_4_layer_call_fn_184389

inputs
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1819732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿD;22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;
 
_user_specified_nameinputs
ó
ù
@__inference_ConC_layer_call_and_return_conditional_losses_180953

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿßÁ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿßÁ
 
_user_specified_nameinputs
Ä
c
*__inference_dropout_8_layer_call_fn_184308

inputs
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_1819042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs

ö
A__inference_HidD1_layer_call_and_return_conditional_losses_181423

inputs3
matmul_readvariableop_resource:á.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:á*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿá: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿá
 
_user_specified_nameinputs
È
C
'__inference_PoolA2_layer_call_fn_183948

inputs
identityæ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolA2_layer_call_and_return_conditional_losses_1807442
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢
^
B__inference_PoolA2_layer_call_and_return_conditional_losses_183938

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢
^
B__inference_PoolB3_layer_call_and_return_conditional_losses_184125

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø

&__inference_HidB2_layer_call_fn_184599

inputs
unknown:¤
	unknown_0:	
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidB2_layer_call_and_return_conditional_losses_1813552
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¤: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
 
_user_specified_nameinputs
Û
C
'__inference_PoolA2_layer_call_fn_183953

inputs
identityË
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolA2_layer_call_and_return_conditional_losses_1810302
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG=:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=
 
_user_specified_nameinputs

ö
A__inference_HidC2_layer_call_and_return_conditional_losses_184650

inputs3
matmul_readvariableop_resource:°.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:°*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ°: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
 
_user_specified_nameinputs
ó

%__inference_Hid1_layer_call_fn_184748

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Hid1_layer_call_and_return_conditional_losses_1814742
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
^
B__inference_PoolB2_layer_call_and_return_conditional_losses_183963

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿF=:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=
 
_user_specified_nameinputs

c
E__inference_dropout_2_layer_call_and_return_conditional_losses_181018

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG>:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>
 
_user_specified_nameinputs
é
a
E__inference_flatten_7_layer_call_and_return_conditional_losses_184472

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ°O  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ":W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
 
_user_specified_nameinputs
Õ
F
*__inference_flatten_1_layer_call_fn_184411

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1812662
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
é
a
E__inference_flatten_6_layer_call_and_return_conditional_losses_181226

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿî 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿî2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿî2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG>:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>
 
_user_specified_nameinputs

c
E__inference_dropout_6_layer_call_and_return_conditional_losses_184232

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
¢
^
B__inference_PoolC2_layer_call_and_return_conditional_losses_184145

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_181950

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG=:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=
 
_user_specified_nameinputs
é
a
E__inference_flatten_7_layer_call_and_return_conditional_losses_181218

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ°O  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ":W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
 
_user_specified_nameinputs
è
ú
A__inference_ConA2_layer_call_and_return_conditional_losses_181007

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿI?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?
 
_user_specified_nameinputs
µ
^
B__inference_PoolB2_layer_call_and_return_conditional_losses_181024

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿF=:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=
 
_user_specified_nameinputs
ó
ù
@__inference_ConE_layer_call_and_return_conditional_losses_184091

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿßÁ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿßÁ
 
_user_specified_nameinputs
µ
^
B__inference_PoolA3_layer_call_and_return_conditional_losses_181145

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ!(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(
 
_user_specified_nameinputs
µ
^
B__inference_PoolC2_layer_call_and_return_conditional_losses_181133

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿE<:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<
 
_user_specified_nameinputs
é
a
E__inference_flatten_6_layer_call_and_return_conditional_losses_184461

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿî 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿî2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿî2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG>:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>
 
_user_specified_nameinputs

ö
A__inference_HidC2_layer_call_and_return_conditional_losses_181406

inputs3
matmul_readvariableop_resource:°.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:°*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ°: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
 
_user_specified_nameinputs
ë
b
C__inference_dropout_layer_call_and_return_conditional_losses_183816

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿI?:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?
 
_user_specified_nameinputs

ö
A__inference_HidB2_layer_call_and_return_conditional_losses_181355

inputs3
matmul_readvariableop_resource:¤.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:¤*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¤: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
 
_user_specified_nameinputs
í
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_182152

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿH?:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?
 
_user_specified_nameinputs


&__inference_ConA2_layer_call_fn_183893

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_ConA2_layer_call_and_return_conditional_losses_1810072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿI?: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?
 
_user_specified_nameinputs
Õ
F
*__inference_flatten_7_layer_call_fn_184477

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_1812182
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ":W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
 
_user_specified_nameinputs
ó

%__inference_Hid2_layer_call_fn_184768

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Hid2_layer_call_and_return_conditional_losses_1814912
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç
a
E__inference_flatten_5_layer_call_and_return_conditional_losses_184450

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ #  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
Ó
F
*__inference_flatten_2_layer_call_fn_184422

inputs
identityÇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_1812582
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
ð
ã
G__inference_concatenate_layer_call_and_return_conditional_losses_184714
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisÒ
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9concat/axis:output:0*
N
*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ý
_input_shapesË
È:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/3:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/4:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/5:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/6:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/7:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/8:R	N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/9
Ù
Ô
$__inference_signature_wrapper_183041	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:	
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:	

unknown_12:$

unknown_13:

unknown_14:$

unknown_15:(

unknown_16:($

unknown_17:(

unknown_18:(

unknown_19:Ê×

unknown_20:	

unknown_21:¤

unknown_22:	

unknown_23:
F

unknown_24:	

unknown_25: ©

unknown_26:	

unknown_27:¤

unknown_28:	

unknown_29:
F

unknown_30:	

unknown_31:î

unknown_32:	

unknown_33:°

unknown_34:	

unknown_35:á

unknown_36:	

unknown_37: ð

unknown_38:	

unknown_39:


unknown_40:	

unknown_41:


unknown_42:	

unknown_43:	(

unknown_44:(
identity¢StatefulPartitionedCall²
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
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_1806692
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes{
y:ÿÿÿÿÿÿÿÿÿßÁ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿßÁ

_user_specified_nameinput
é
a
E__inference_flatten_4_layer_call_and_return_conditional_losses_181242

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿR  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
¡
]
A__inference_PoolD_layer_call_and_return_conditional_losses_180854

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

õ
@__inference_HidA_layer_call_and_return_conditional_losses_184510

inputs3
matmul_readvariableop_resource:Ê×.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:Ê×*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿÊ×: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ×
 
_user_specified_nameinputs

ô
@__inference_Hid2_layer_call_and_return_conditional_losses_184759

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanhd
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³á
ë
A__inference_model_layer_call_and_return_conditional_losses_182936	
input%
conb_182789:
conb_182791:%
cona_182794:
cona_182796:%
conc_182801:	
conc_182803:&
conb2_182809:
conb2_182811:&
cona2_182814:
cona2_182816:%
cone_182822:
cone_182824:%
cond_182827:	
cond_182829:&
conc2_182832:
conc2_182834:&
conb3_182837:(
conb3_182839:(&
cona3_182842:(
cona3_182844:( 
hida_182869:Ê×
hida_182871:	!
hida2_182874:¤
hida2_182876:	 
hida3_182879:
F
hida3_182881:	 
hidb_182884: ©
hidb_182886:	!
hidb2_182889:¤
hidb2_182891:	 
hidb3_182894:
F
hidb3_182896:	 
hidc_182899:î
hidc_182901:	!
hidc2_182904:°
hidc2_182906:	!
hidd1_182909:á
hidd1_182911:	!
hide1_182914: ð
hide1_182916:	
hid1_182920:

hid1_182922:	
hid2_182925:

hid2_182927:	 
output_182930:	(
output_182932:(
identity¢ConA/StatefulPartitionedCall¢ConA2/StatefulPartitionedCall¢ConA3/StatefulPartitionedCall¢ConB/StatefulPartitionedCall¢ConB2/StatefulPartitionedCall¢ConB3/StatefulPartitionedCall¢ConC/StatefulPartitionedCall¢ConC2/StatefulPartitionedCall¢ConD/StatefulPartitionedCall¢ConE/StatefulPartitionedCall¢Hid1/StatefulPartitionedCall¢Hid2/StatefulPartitionedCall¢HidA/StatefulPartitionedCall¢HidA2/StatefulPartitionedCall¢HidA3/StatefulPartitionedCall¢HidB/StatefulPartitionedCall¢HidB2/StatefulPartitionedCall¢HidB3/StatefulPartitionedCall¢HidC/StatefulPartitionedCall¢HidC2/StatefulPartitionedCall¢HidD1/StatefulPartitionedCall¢HidE1/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢!dropout_4/StatefulPartitionedCall¢!dropout_5/StatefulPartitionedCall¢!dropout_6/StatefulPartitionedCall¢!dropout_7/StatefulPartitionedCall¢!dropout_8/StatefulPartitionedCall¢!dropout_9/StatefulPartitionedCall¢output/StatefulPartitionedCall
ConB/StatefulPartitionedCallStatefulPartitionedCallinputconb_182789conb_182791*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ConB_layer_call_and_return_conditional_losses_1809072
ConB/StatefulPartitionedCall
ConA/StatefulPartitionedCallStatefulPartitionedCallinputcona_182794cona_182796*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ConA_layer_call_and_return_conditional_losses_1809242
ConA/StatefulPartitionedCallõ
PoolB/PartitionedCallPartitionedCall%ConB/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolB_layer_call_and_return_conditional_losses_1809342
PoolB/PartitionedCallõ
PoolA/PartitionedCallPartitionedCall%ConA/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolA_layer_call_and_return_conditional_losses_1809402
PoolA/PartitionedCall
ConC/StatefulPartitionedCallStatefulPartitionedCallinputconc_182801conc_182803*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ConC_layer_call_and_return_conditional_losses_1809532
ConC/StatefulPartitionedCall
!dropout_1/StatefulPartitionedCallStatefulPartitionedCallPoolB/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1821522#
!dropout_1/StatefulPartitionedCall°
dropout/StatefulPartitionedCallStatefulPartitionedCallPoolA/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1821292!
dropout/StatefulPartitionedCallõ
PoolC/PartitionedCallPartitionedCall%ConC/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolC_layer_call_and_return_conditional_losses_1809772
PoolC/PartitionedCall´
ConB2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conb2_182809conb2_182811*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_ConB2_layer_call_and_return_conditional_losses_1809902
ConB2/StatefulPartitionedCall²
ConA2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0cona2_182814cona2_182816*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_ConA2_layer_call_and_return_conditional_losses_1810072
ConA2/StatefulPartitionedCall´
!dropout_2/StatefulPartitionedCallStatefulPartitionedCallPoolC/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_1820812#
!dropout_2/StatefulPartitionedCallù
PoolB2/PartitionedCallPartitionedCall&ConB2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolB2_layer_call_and_return_conditional_losses_1810242
PoolB2/PartitionedCallù
PoolA2/PartitionedCallPartitionedCall&ConA2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolA2_layer_call_and_return_conditional_losses_1810302
PoolA2/PartitionedCall
ConE/StatefulPartitionedCallStatefulPartitionedCallinputcone_182822cone_182824*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ConE_layer_call_and_return_conditional_losses_1810432
ConE/StatefulPartitionedCall
ConD/StatefulPartitionedCallStatefulPartitionedCallinputcond_182827cond_182829*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ConD_layer_call_and_return_conditional_losses_1810602
ConD/StatefulPartitionedCall´
ConC2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0conc2_182832conc2_182834*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_ConC2_layer_call_and_return_conditional_losses_1810772
ConC2/StatefulPartitionedCall©
ConB3/StatefulPartitionedCallStatefulPartitionedCallPoolB2/PartitionedCall:output:0conb3_182837conb3_182839*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_ConB3_layer_call_and_return_conditional_losses_1810942
ConB3/StatefulPartitionedCall©
ConA3/StatefulPartitionedCallStatefulPartitionedCallPoolA2/PartitionedCall:output:0cona3_182842cona3_182844*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_ConA3_layer_call_and_return_conditional_losses_1811112
ConA3/StatefulPartitionedCallõ
PoolE/PartitionedCallPartitionedCall%ConE/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolE_layer_call_and_return_conditional_losses_1811212
PoolE/PartitionedCallõ
PoolD/PartitionedCallPartitionedCall%ConD/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolD_layer_call_and_return_conditional_losses_1811272
PoolD/PartitionedCallù
PoolC2/PartitionedCallPartitionedCall&ConC2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolC2_layer_call_and_return_conditional_losses_1811332
PoolC2/PartitionedCallù
PoolB3/PartitionedCallPartitionedCall&ConB3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolB3_layer_call_and_return_conditional_losses_1811392
PoolB3/PartitionedCallù
PoolA3/PartitionedCallPartitionedCall&ConA3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolA3_layer_call_and_return_conditional_losses_1811452
PoolA3/PartitionedCall¶
!dropout_4/StatefulPartitionedCallStatefulPartitionedCallPoolE/PartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1819732#
!dropout_4/StatefulPartitionedCall¶
!dropout_3/StatefulPartitionedCallStatefulPartitionedCallPoolD/PartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_1819502#
!dropout_3/StatefulPartitionedCall·
!dropout_9/StatefulPartitionedCallStatefulPartitionedCallPoolC2/PartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_1819272#
!dropout_9/StatefulPartitionedCall·
!dropout_8/StatefulPartitionedCallStatefulPartitionedCallPoolB3/PartitionedCall:output:0"^dropout_9/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_1819042#
!dropout_8/StatefulPartitionedCall·
!dropout_7/StatefulPartitionedCallStatefulPartitionedCallPoolB2/PartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_1818812#
!dropout_7/StatefulPartitionedCall·
!dropout_6/StatefulPartitionedCallStatefulPartitionedCallPoolA3/PartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_1818582#
!dropout_6/StatefulPartitionedCall·
!dropout_5/StatefulPartitionedCallStatefulPartitionedCallPoolA2/PartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_1818352#
!dropout_5/StatefulPartitionedCall
flatten_9/PartitionedCallPartitionedCall*dropout_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ð* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_9_layer_call_and_return_conditional_losses_1812022
flatten_9/PartitionedCall
flatten_8/PartitionedCallPartitionedCall*dropout_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿá* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_1812102
flatten_8/PartitionedCall
flatten_7/PartitionedCallPartitionedCall*dropout_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_1812182
flatten_7/PartitionedCall
flatten_6/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿî* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_1812262
flatten_6/PartitionedCallÿ
flatten_5/PartitionedCallPartitionedCall*dropout_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_1812342
flatten_5/PartitionedCall
flatten_4/PartitionedCallPartitionedCall*dropout_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_1812422
flatten_4/PartitionedCallô
flatten_3/PartitionedCallPartitionedCallPoolB/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1812502
flatten_3/PartitionedCallÿ
flatten_2/PartitionedCallPartitionedCall*dropout_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_1812582
flatten_2/PartitionedCall
flatten_1/PartitionedCallPartitionedCall*dropout_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1812662
flatten_1/PartitionedCallî
flatten/PartitionedCallPartitionedCallPoolA/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ×* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1812742
flatten/PartitionedCall
HidA/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0hida_182869hida_182871*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_HidA_layer_call_and_return_conditional_losses_1812872
HidA/StatefulPartitionedCall¥
HidA2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0hida2_182874hida2_182876*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidA2_layer_call_and_return_conditional_losses_1813042
HidA2/StatefulPartitionedCall¥
HidA3/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0hida3_182879hida3_182881*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidA3_layer_call_and_return_conditional_losses_1813212
HidA3/StatefulPartitionedCall 
HidB/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0hidb_182884hidb_182886*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_HidB_layer_call_and_return_conditional_losses_1813382
HidB/StatefulPartitionedCall¥
HidB2/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0hidb2_182889hidb2_182891*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidB2_layer_call_and_return_conditional_losses_1813552
HidB2/StatefulPartitionedCall¥
HidB3/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0hidb3_182894hidb3_182896*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidB3_layer_call_and_return_conditional_losses_1813722
HidB3/StatefulPartitionedCall 
HidC/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0hidc_182899hidc_182901*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_HidC_layer_call_and_return_conditional_losses_1813892
HidC/StatefulPartitionedCall¥
HidC2/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0hidc2_182904hidc2_182906*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidC2_layer_call_and_return_conditional_losses_1814062
HidC2/StatefulPartitionedCall¥
HidD1/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0hidd1_182909hidd1_182911*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidD1_layer_call_and_return_conditional_losses_1814232
HidD1/StatefulPartitionedCall¥
HidE1/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0hide1_182914hide1_182916*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidE1_layer_call_and_return_conditional_losses_1814402
HidE1/StatefulPartitionedCallï
concatenate/PartitionedCallPartitionedCall%HidA/StatefulPartitionedCall:output:0&HidA2/StatefulPartitionedCall:output:0&HidA3/StatefulPartitionedCall:output:0%HidB/StatefulPartitionedCall:output:0&HidB2/StatefulPartitionedCall:output:0&HidB3/StatefulPartitionedCall:output:0%HidC/StatefulPartitionedCall:output:0&HidC2/StatefulPartitionedCall:output:0&HidD1/StatefulPartitionedCall:output:0&HidE1/StatefulPartitionedCall:output:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_1814612
concatenate/PartitionedCall¢
Hid1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0hid1_182920hid1_182922*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Hid1_layer_call_and_return_conditional_losses_1814742
Hid1/StatefulPartitionedCall£
Hid2/StatefulPartitionedCallStatefulPartitionedCall%Hid1/StatefulPartitionedCall:output:0hid2_182925hid2_182927*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Hid2_layer_call_and_return_conditional_losses_1814912
Hid2/StatefulPartitionedCall¬
output/StatefulPartitionedCallStatefulPartitionedCall%Hid2/StatefulPartitionedCall:output:0output_182930output_182932*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1815082 
output/StatefulPartitionedCall
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity	
NoOpNoOp^ConA/StatefulPartitionedCall^ConA2/StatefulPartitionedCall^ConA3/StatefulPartitionedCall^ConB/StatefulPartitionedCall^ConB2/StatefulPartitionedCall^ConB3/StatefulPartitionedCall^ConC/StatefulPartitionedCall^ConC2/StatefulPartitionedCall^ConD/StatefulPartitionedCall^ConE/StatefulPartitionedCall^Hid1/StatefulPartitionedCall^Hid2/StatefulPartitionedCall^HidA/StatefulPartitionedCall^HidA2/StatefulPartitionedCall^HidA3/StatefulPartitionedCall^HidB/StatefulPartitionedCall^HidB2/StatefulPartitionedCall^HidB3/StatefulPartitionedCall^HidC/StatefulPartitionedCall^HidC2/StatefulPartitionedCall^HidD1/StatefulPartitionedCall^HidE1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes{
y:ÿÿÿÿÿÿÿÿÿßÁ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
ConA/StatefulPartitionedCallConA/StatefulPartitionedCall2>
ConA2/StatefulPartitionedCallConA2/StatefulPartitionedCall2>
ConA3/StatefulPartitionedCallConA3/StatefulPartitionedCall2<
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
HidA3/StatefulPartitionedCallHidA3/StatefulPartitionedCall2<
HidB/StatefulPartitionedCallHidB/StatefulPartitionedCall2>
HidB2/StatefulPartitionedCallHidB2/StatefulPartitionedCall2>
HidB3/StatefulPartitionedCallHidB3/StatefulPartitionedCall2<
HidC/StatefulPartitionedCallHidC/StatefulPartitionedCall2>
HidC2/StatefulPartitionedCallHidC2/StatefulPartitionedCall2>
HidD1/StatefulPartitionedCallHidD1/StatefulPartitionedCall2>
HidE1/StatefulPartitionedCallHidE1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
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
:ÿÿÿÿÿÿÿÿÿßÁ

_user_specified_nameinput
é
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_184428

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ Ô  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿH?:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?
 
_user_specified_nameinputs

õ
A__inference_HidA3_layer_call_and_return_conditional_losses_181321

inputs2
matmul_readvariableop_resource:
F.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
F*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿF: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
Ý
B
&__inference_PoolB_layer_call_fn_183799

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolB_layer_call_and_return_conditional_losses_1809342
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÙ½:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½
 
_user_specified_nameinputs
é
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_184406

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿR  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
µ
^
B__inference_PoolA3_layer_call_and_return_conditional_losses_184110

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ!(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(
 
_user_specified_nameinputs
è
ú
A__inference_ConA3_layer_call_and_return_conditional_losses_184011

inputs8
conv2d_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs

c
E__inference_dropout_9_layer_call_and_return_conditional_losses_181166

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ":W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
 
_user_specified_nameinputs
Ü
È
,__inference_concatenate_layer_call_fn_184728
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
identity®
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9*
Tin
2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_1814612
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ý
_input_shapesË
È:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/3:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/4:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/5:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/6:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/7:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/8:R	N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/9
Ø
á
G__inference_concatenate_layer_call_and_return_conditional_losses_181461

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisÐ
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9concat/axis:output:0*
N
*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ý
_input_shapesË
È:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:P	L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
d
E__inference_dropout_7_layer_call_and_return_conditional_losses_184271

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
Õ
F
*__inference_flatten_3_layer_call_fn_184433

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1812502
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿH?:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?
 
_user_specified_nameinputs
ç
_
C__inference_flatten_layer_call_and_return_conditional_losses_184395

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÊk  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ×2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ×2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿI?:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?
 
_user_specified_nameinputs
é
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_181266

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿR  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs

õ
A__inference_HidA3_layer_call_and_return_conditional_losses_184550

inputs2
matmul_readvariableop_resource:
F.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
F*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿF: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
í
d
E__inference_dropout_8_layer_call_and_return_conditional_losses_184298

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs

õ
@__inference_HidC_layer_call_and_return_conditional_losses_184630

inputs3
matmul_readvariableop_resource:î.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:î*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿî: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿî
 
_user_specified_nameinputs
í
d
E__inference_dropout_5_layer_call_and_return_conditional_losses_184217

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
ºá
ì
A__inference_model_layer_call_and_return_conditional_losses_182444

inputs%
conb_182297:
conb_182299:%
cona_182302:
cona_182304:%
conc_182309:	
conc_182311:&
conb2_182317:
conb2_182319:&
cona2_182322:
cona2_182324:%
cone_182330:
cone_182332:%
cond_182335:	
cond_182337:&
conc2_182340:
conc2_182342:&
conb3_182345:(
conb3_182347:(&
cona3_182350:(
cona3_182352:( 
hida_182377:Ê×
hida_182379:	!
hida2_182382:¤
hida2_182384:	 
hida3_182387:
F
hida3_182389:	 
hidb_182392: ©
hidb_182394:	!
hidb2_182397:¤
hidb2_182399:	 
hidb3_182402:
F
hidb3_182404:	 
hidc_182407:î
hidc_182409:	!
hidc2_182412:°
hidc2_182414:	!
hidd1_182417:á
hidd1_182419:	!
hide1_182422: ð
hide1_182424:	
hid1_182428:

hid1_182430:	
hid2_182433:

hid2_182435:	 
output_182438:	(
output_182440:(
identity¢ConA/StatefulPartitionedCall¢ConA2/StatefulPartitionedCall¢ConA3/StatefulPartitionedCall¢ConB/StatefulPartitionedCall¢ConB2/StatefulPartitionedCall¢ConB3/StatefulPartitionedCall¢ConC/StatefulPartitionedCall¢ConC2/StatefulPartitionedCall¢ConD/StatefulPartitionedCall¢ConE/StatefulPartitionedCall¢Hid1/StatefulPartitionedCall¢Hid2/StatefulPartitionedCall¢HidA/StatefulPartitionedCall¢HidA2/StatefulPartitionedCall¢HidA3/StatefulPartitionedCall¢HidB/StatefulPartitionedCall¢HidB2/StatefulPartitionedCall¢HidB3/StatefulPartitionedCall¢HidC/StatefulPartitionedCall¢HidC2/StatefulPartitionedCall¢HidD1/StatefulPartitionedCall¢HidE1/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢!dropout_4/StatefulPartitionedCall¢!dropout_5/StatefulPartitionedCall¢!dropout_6/StatefulPartitionedCall¢!dropout_7/StatefulPartitionedCall¢!dropout_8/StatefulPartitionedCall¢!dropout_9/StatefulPartitionedCall¢output/StatefulPartitionedCall
ConB/StatefulPartitionedCallStatefulPartitionedCallinputsconb_182297conb_182299*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ConB_layer_call_and_return_conditional_losses_1809072
ConB/StatefulPartitionedCall
ConA/StatefulPartitionedCallStatefulPartitionedCallinputscona_182302cona_182304*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ConA_layer_call_and_return_conditional_losses_1809242
ConA/StatefulPartitionedCallõ
PoolB/PartitionedCallPartitionedCall%ConB/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolB_layer_call_and_return_conditional_losses_1809342
PoolB/PartitionedCallõ
PoolA/PartitionedCallPartitionedCall%ConA/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolA_layer_call_and_return_conditional_losses_1809402
PoolA/PartitionedCall
ConC/StatefulPartitionedCallStatefulPartitionedCallinputsconc_182309conc_182311*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ConC_layer_call_and_return_conditional_losses_1809532
ConC/StatefulPartitionedCall
!dropout_1/StatefulPartitionedCallStatefulPartitionedCallPoolB/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1821522#
!dropout_1/StatefulPartitionedCall°
dropout/StatefulPartitionedCallStatefulPartitionedCallPoolA/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1821292!
dropout/StatefulPartitionedCallõ
PoolC/PartitionedCallPartitionedCall%ConC/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolC_layer_call_and_return_conditional_losses_1809772
PoolC/PartitionedCall´
ConB2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conb2_182317conb2_182319*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_ConB2_layer_call_and_return_conditional_losses_1809902
ConB2/StatefulPartitionedCall²
ConA2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0cona2_182322cona2_182324*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_ConA2_layer_call_and_return_conditional_losses_1810072
ConA2/StatefulPartitionedCall´
!dropout_2/StatefulPartitionedCallStatefulPartitionedCallPoolC/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_1820812#
!dropout_2/StatefulPartitionedCallù
PoolB2/PartitionedCallPartitionedCall&ConB2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolB2_layer_call_and_return_conditional_losses_1810242
PoolB2/PartitionedCallù
PoolA2/PartitionedCallPartitionedCall&ConA2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolA2_layer_call_and_return_conditional_losses_1810302
PoolA2/PartitionedCall
ConE/StatefulPartitionedCallStatefulPartitionedCallinputscone_182330cone_182332*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ConE_layer_call_and_return_conditional_losses_1810432
ConE/StatefulPartitionedCall
ConD/StatefulPartitionedCallStatefulPartitionedCallinputscond_182335cond_182337*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ConD_layer_call_and_return_conditional_losses_1810602
ConD/StatefulPartitionedCall´
ConC2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0conc2_182340conc2_182342*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_ConC2_layer_call_and_return_conditional_losses_1810772
ConC2/StatefulPartitionedCall©
ConB3/StatefulPartitionedCallStatefulPartitionedCallPoolB2/PartitionedCall:output:0conb3_182345conb3_182347*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_ConB3_layer_call_and_return_conditional_losses_1810942
ConB3/StatefulPartitionedCall©
ConA3/StatefulPartitionedCallStatefulPartitionedCallPoolA2/PartitionedCall:output:0cona3_182350cona3_182352*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_ConA3_layer_call_and_return_conditional_losses_1811112
ConA3/StatefulPartitionedCallõ
PoolE/PartitionedCallPartitionedCall%ConE/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolE_layer_call_and_return_conditional_losses_1811212
PoolE/PartitionedCallõ
PoolD/PartitionedCallPartitionedCall%ConD/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolD_layer_call_and_return_conditional_losses_1811272
PoolD/PartitionedCallù
PoolC2/PartitionedCallPartitionedCall&ConC2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolC2_layer_call_and_return_conditional_losses_1811332
PoolC2/PartitionedCallù
PoolB3/PartitionedCallPartitionedCall&ConB3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolB3_layer_call_and_return_conditional_losses_1811392
PoolB3/PartitionedCallù
PoolA3/PartitionedCallPartitionedCall&ConA3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolA3_layer_call_and_return_conditional_losses_1811452
PoolA3/PartitionedCall¶
!dropout_4/StatefulPartitionedCallStatefulPartitionedCallPoolE/PartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1819732#
!dropout_4/StatefulPartitionedCall¶
!dropout_3/StatefulPartitionedCallStatefulPartitionedCallPoolD/PartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_1819502#
!dropout_3/StatefulPartitionedCall·
!dropout_9/StatefulPartitionedCallStatefulPartitionedCallPoolC2/PartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_1819272#
!dropout_9/StatefulPartitionedCall·
!dropout_8/StatefulPartitionedCallStatefulPartitionedCallPoolB3/PartitionedCall:output:0"^dropout_9/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_1819042#
!dropout_8/StatefulPartitionedCall·
!dropout_7/StatefulPartitionedCallStatefulPartitionedCallPoolB2/PartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_1818812#
!dropout_7/StatefulPartitionedCall·
!dropout_6/StatefulPartitionedCallStatefulPartitionedCallPoolA3/PartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_1818582#
!dropout_6/StatefulPartitionedCall·
!dropout_5/StatefulPartitionedCallStatefulPartitionedCallPoolA2/PartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_1818352#
!dropout_5/StatefulPartitionedCall
flatten_9/PartitionedCallPartitionedCall*dropout_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ð* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_9_layer_call_and_return_conditional_losses_1812022
flatten_9/PartitionedCall
flatten_8/PartitionedCallPartitionedCall*dropout_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿá* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_1812102
flatten_8/PartitionedCall
flatten_7/PartitionedCallPartitionedCall*dropout_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_1812182
flatten_7/PartitionedCall
flatten_6/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿî* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_1812262
flatten_6/PartitionedCallÿ
flatten_5/PartitionedCallPartitionedCall*dropout_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_1812342
flatten_5/PartitionedCall
flatten_4/PartitionedCallPartitionedCall*dropout_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_1812422
flatten_4/PartitionedCallô
flatten_3/PartitionedCallPartitionedCallPoolB/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1812502
flatten_3/PartitionedCallÿ
flatten_2/PartitionedCallPartitionedCall*dropout_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_1812582
flatten_2/PartitionedCall
flatten_1/PartitionedCallPartitionedCall*dropout_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1812662
flatten_1/PartitionedCallî
flatten/PartitionedCallPartitionedCallPoolA/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ×* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1812742
flatten/PartitionedCall
HidA/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0hida_182377hida_182379*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_HidA_layer_call_and_return_conditional_losses_1812872
HidA/StatefulPartitionedCall¥
HidA2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0hida2_182382hida2_182384*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidA2_layer_call_and_return_conditional_losses_1813042
HidA2/StatefulPartitionedCall¥
HidA3/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0hida3_182387hida3_182389*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidA3_layer_call_and_return_conditional_losses_1813212
HidA3/StatefulPartitionedCall 
HidB/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0hidb_182392hidb_182394*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_HidB_layer_call_and_return_conditional_losses_1813382
HidB/StatefulPartitionedCall¥
HidB2/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0hidb2_182397hidb2_182399*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidB2_layer_call_and_return_conditional_losses_1813552
HidB2/StatefulPartitionedCall¥
HidB3/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0hidb3_182402hidb3_182404*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidB3_layer_call_and_return_conditional_losses_1813722
HidB3/StatefulPartitionedCall 
HidC/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0hidc_182407hidc_182409*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_HidC_layer_call_and_return_conditional_losses_1813892
HidC/StatefulPartitionedCall¥
HidC2/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0hidc2_182412hidc2_182414*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidC2_layer_call_and_return_conditional_losses_1814062
HidC2/StatefulPartitionedCall¥
HidD1/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0hidd1_182417hidd1_182419*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidD1_layer_call_and_return_conditional_losses_1814232
HidD1/StatefulPartitionedCall¥
HidE1/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0hide1_182422hide1_182424*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidE1_layer_call_and_return_conditional_losses_1814402
HidE1/StatefulPartitionedCallï
concatenate/PartitionedCallPartitionedCall%HidA/StatefulPartitionedCall:output:0&HidA2/StatefulPartitionedCall:output:0&HidA3/StatefulPartitionedCall:output:0%HidB/StatefulPartitionedCall:output:0&HidB2/StatefulPartitionedCall:output:0&HidB3/StatefulPartitionedCall:output:0%HidC/StatefulPartitionedCall:output:0&HidC2/StatefulPartitionedCall:output:0&HidD1/StatefulPartitionedCall:output:0&HidE1/StatefulPartitionedCall:output:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_1814612
concatenate/PartitionedCall¢
Hid1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0hid1_182428hid1_182430*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Hid1_layer_call_and_return_conditional_losses_1814742
Hid1/StatefulPartitionedCall£
Hid2/StatefulPartitionedCallStatefulPartitionedCall%Hid1/StatefulPartitionedCall:output:0hid2_182433hid2_182435*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Hid2_layer_call_and_return_conditional_losses_1814912
Hid2/StatefulPartitionedCall¬
output/StatefulPartitionedCallStatefulPartitionedCall%Hid2/StatefulPartitionedCall:output:0output_182438output_182440*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1815082 
output/StatefulPartitionedCall
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity	
NoOpNoOp^ConA/StatefulPartitionedCall^ConA2/StatefulPartitionedCall^ConA3/StatefulPartitionedCall^ConB/StatefulPartitionedCall^ConB2/StatefulPartitionedCall^ConB3/StatefulPartitionedCall^ConC/StatefulPartitionedCall^ConC2/StatefulPartitionedCall^ConD/StatefulPartitionedCall^ConE/StatefulPartitionedCall^Hid1/StatefulPartitionedCall^Hid2/StatefulPartitionedCall^HidA/StatefulPartitionedCall^HidA2/StatefulPartitionedCall^HidA3/StatefulPartitionedCall^HidB/StatefulPartitionedCall^HidB2/StatefulPartitionedCall^HidB3/StatefulPartitionedCall^HidC/StatefulPartitionedCall^HidC2/StatefulPartitionedCall^HidD1/StatefulPartitionedCall^HidE1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes{
y:ÿÿÿÿÿÿÿÿÿßÁ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
ConA/StatefulPartitionedCallConA/StatefulPartitionedCall2>
ConA2/StatefulPartitionedCallConA2/StatefulPartitionedCall2>
ConA3/StatefulPartitionedCallConA3/StatefulPartitionedCall2<
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
HidA3/StatefulPartitionedCallHidA3/StatefulPartitionedCall2<
HidB/StatefulPartitionedCallHidB/StatefulPartitionedCall2>
HidB2/StatefulPartitionedCallHidB2/StatefulPartitionedCall2>
HidB3/StatefulPartitionedCallHidB3/StatefulPartitionedCall2<
HidC/StatefulPartitionedCallHidC/StatefulPartitionedCall2>
HidC2/StatefulPartitionedCallHidC2/StatefulPartitionedCall2>
HidD1/StatefulPartitionedCallHidD1/StatefulPartitionedCall2>
HidE1/StatefulPartitionedCallHidE1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
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
:ÿÿÿÿÿÿÿÿÿßÁ
 
_user_specified_nameinputs
þ
×
&__inference_model_layer_call_fn_183719

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:	
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:	

unknown_12:$

unknown_13:

unknown_14:$

unknown_15:(

unknown_16:($

unknown_17:(

unknown_18:(

unknown_19:Ê×

unknown_20:	

unknown_21:¤

unknown_22:	

unknown_23:
F

unknown_24:	

unknown_25: ©

unknown_26:	

unknown_27:¤

unknown_28:	

unknown_29:
F

unknown_30:	

unknown_31:î

unknown_32:	

unknown_33:°

unknown_34:	

unknown_35:á

unknown_36:	

unknown_37: ð

unknown_38:	

unknown_39:


unknown_40:	

unknown_41:


unknown_42:	

unknown_43:	(

unknown_44:(
identity¢StatefulPartitionedCallÓ
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
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1824442
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes{
y:ÿÿÿÿÿÿÿÿÿßÁ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿßÁ
 
_user_specified_nameinputs


%__inference_ConD_layer_call_fn_184080

inputs!
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ConD_layer_call_and_return_conditional_losses_1810602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿßÁ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿßÁ
 
_user_specified_nameinputs
Ý
B
&__inference_PoolD_layer_call_fn_184180

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolD_layer_call_and_return_conditional_losses_1811272
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹
 
_user_specified_nameinputs
ó

'__inference_output_layer_call_fn_184788

inputs
unknown:	(
	unknown_0:(
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1815082
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢
^
B__inference_PoolA2_layer_call_and_return_conditional_losses_180744

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ

A__inference_model_layer_call_and_return_conditional_losses_182786	
input%
conb_182639:
conb_182641:%
cona_182644:
cona_182646:%
conc_182651:	
conc_182653:&
conb2_182659:
conb2_182661:&
cona2_182664:
cona2_182666:%
cone_182672:
cone_182674:%
cond_182677:	
cond_182679:&
conc2_182682:
conc2_182684:&
conb3_182687:(
conb3_182689:(&
cona3_182692:(
cona3_182694:( 
hida_182719:Ê×
hida_182721:	!
hida2_182724:¤
hida2_182726:	 
hida3_182729:
F
hida3_182731:	 
hidb_182734: ©
hidb_182736:	!
hidb2_182739:¤
hidb2_182741:	 
hidb3_182744:
F
hidb3_182746:	 
hidc_182749:î
hidc_182751:	!
hidc2_182754:°
hidc2_182756:	!
hidd1_182759:á
hidd1_182761:	!
hide1_182764: ð
hide1_182766:	
hid1_182770:

hid1_182772:	
hid2_182775:

hid2_182777:	 
output_182780:	(
output_182782:(
identity¢ConA/StatefulPartitionedCall¢ConA2/StatefulPartitionedCall¢ConA3/StatefulPartitionedCall¢ConB/StatefulPartitionedCall¢ConB2/StatefulPartitionedCall¢ConB3/StatefulPartitionedCall¢ConC/StatefulPartitionedCall¢ConC2/StatefulPartitionedCall¢ConD/StatefulPartitionedCall¢ConE/StatefulPartitionedCall¢Hid1/StatefulPartitionedCall¢Hid2/StatefulPartitionedCall¢HidA/StatefulPartitionedCall¢HidA2/StatefulPartitionedCall¢HidA3/StatefulPartitionedCall¢HidB/StatefulPartitionedCall¢HidB2/StatefulPartitionedCall¢HidB3/StatefulPartitionedCall¢HidC/StatefulPartitionedCall¢HidC2/StatefulPartitionedCall¢HidD1/StatefulPartitionedCall¢HidE1/StatefulPartitionedCall¢output/StatefulPartitionedCall
ConB/StatefulPartitionedCallStatefulPartitionedCallinputconb_182639conb_182641*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ½*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ConB_layer_call_and_return_conditional_losses_1809072
ConB/StatefulPartitionedCall
ConA/StatefulPartitionedCallStatefulPartitionedCallinputcona_182644cona_182646*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ¿*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ConA_layer_call_and_return_conditional_losses_1809242
ConA/StatefulPartitionedCallõ
PoolB/PartitionedCallPartitionedCall%ConB/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolB_layer_call_and_return_conditional_losses_1809342
PoolB/PartitionedCallõ
PoolA/PartitionedCallPartitionedCall%ConA/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolA_layer_call_and_return_conditional_losses_1809402
PoolA/PartitionedCall
ConC/StatefulPartitionedCallStatefulPartitionedCallinputconc_182651conc_182653*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×»*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ConC_layer_call_and_return_conditional_losses_1809532
ConC/StatefulPartitionedCallú
dropout_1/PartitionedCallPartitionedCallPoolB/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1809642
dropout_1/PartitionedCallô
dropout/PartitionedCallPartitionedCallPoolA/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿI?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1809712
dropout/PartitionedCallõ
PoolC/PartitionedCallPartitionedCall%ConC/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolC_layer_call_and_return_conditional_losses_1809772
PoolC/PartitionedCall¬
ConB2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conb2_182659conb2_182661*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿF=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_ConB2_layer_call_and_return_conditional_losses_1809902
ConB2/StatefulPartitionedCallª
ConA2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0cona2_182664cona2_182666*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_ConA2_layer_call_and_return_conditional_losses_1810072
ConA2/StatefulPartitionedCallú
dropout_2/PartitionedCallPartitionedCallPoolC/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_1810182
dropout_2/PartitionedCallù
PoolB2/PartitionedCallPartitionedCall&ConB2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolB2_layer_call_and_return_conditional_losses_1810242
PoolB2/PartitionedCallù
PoolA2/PartitionedCallPartitionedCall&ConA2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolA2_layer_call_and_return_conditional_losses_1810302
PoolA2/PartitionedCall
ConE/StatefulPartitionedCallStatefulPartitionedCallinputcone_182672cone_182674*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ³*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ConE_layer_call_and_return_conditional_losses_1810432
ConE/StatefulPartitionedCall
ConD/StatefulPartitionedCallStatefulPartitionedCallinputcond_182677cond_182679*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ¹*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ConD_layer_call_and_return_conditional_losses_1810602
ConD/StatefulPartitionedCall¬
ConC2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0conc2_182682conc2_182684*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_ConC2_layer_call_and_return_conditional_losses_1810772
ConC2/StatefulPartitionedCall©
ConB3/StatefulPartitionedCallStatefulPartitionedCallPoolB2/PartitionedCall:output:0conb3_182687conb3_182689*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_ConB3_layer_call_and_return_conditional_losses_1810942
ConB3/StatefulPartitionedCall©
ConA3/StatefulPartitionedCallStatefulPartitionedCallPoolA2/PartitionedCall:output:0cona3_182692cona3_182694*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_ConA3_layer_call_and_return_conditional_losses_1811112
ConA3/StatefulPartitionedCallõ
PoolE/PartitionedCallPartitionedCall%ConE/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolE_layer_call_and_return_conditional_losses_1811212
PoolE/PartitionedCallõ
PoolD/PartitionedCallPartitionedCall%ConD/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolD_layer_call_and_return_conditional_losses_1811272
PoolD/PartitionedCallù
PoolC2/PartitionedCallPartitionedCall&ConC2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolC2_layer_call_and_return_conditional_losses_1811332
PoolC2/PartitionedCallù
PoolB3/PartitionedCallPartitionedCall&ConB3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolB3_layer_call_and_return_conditional_losses_1811392
PoolB3/PartitionedCallù
PoolA3/PartitionedCallPartitionedCall&ConA3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolA3_layer_call_and_return_conditional_losses_1811452
PoolA3/PartitionedCallú
dropout_4/PartitionedCallPartitionedCallPoolE/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿD;* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1811522
dropout_4/PartitionedCallú
dropout_3/PartitionedCallPartitionedCallPoolD/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_1811592
dropout_3/PartitionedCallû
dropout_9/PartitionedCallPartitionedCallPoolC2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_1811662
dropout_9/PartitionedCallû
dropout_8/PartitionedCallPartitionedCallPoolB3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_1811732
dropout_8/PartitionedCallû
dropout_7/PartitionedCallPartitionedCallPoolB2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_1811802
dropout_7/PartitionedCallû
dropout_6/PartitionedCallPartitionedCallPoolA3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_1811872
dropout_6/PartitionedCallû
dropout_5/PartitionedCallPartitionedCallPoolA2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_1811942
dropout_5/PartitionedCallø
flatten_9/PartitionedCallPartitionedCall"dropout_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ð* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_9_layer_call_and_return_conditional_losses_1812022
flatten_9/PartitionedCallø
flatten_8/PartitionedCallPartitionedCall"dropout_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿá* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_1812102
flatten_8/PartitionedCallø
flatten_7/PartitionedCallPartitionedCall"dropout_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_1812182
flatten_7/PartitionedCallø
flatten_6/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿî* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_1812262
flatten_6/PartitionedCall÷
flatten_5/PartitionedCallPartitionedCall"dropout_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_1812342
flatten_5/PartitionedCallø
flatten_4/PartitionedCallPartitionedCall"dropout_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_1812422
flatten_4/PartitionedCallô
flatten_3/PartitionedCallPartitionedCallPoolB/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1812502
flatten_3/PartitionedCall÷
flatten_2/PartitionedCallPartitionedCall"dropout_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_1812582
flatten_2/PartitionedCallø
flatten_1/PartitionedCallPartitionedCall"dropout_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1812662
flatten_1/PartitionedCallî
flatten/PartitionedCallPartitionedCallPoolA/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ×* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1812742
flatten/PartitionedCall
HidA/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0hida_182719hida_182721*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_HidA_layer_call_and_return_conditional_losses_1812872
HidA/StatefulPartitionedCall¥
HidA2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0hida2_182724hida2_182726*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidA2_layer_call_and_return_conditional_losses_1813042
HidA2/StatefulPartitionedCall¥
HidA3/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0hida3_182729hida3_182731*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidA3_layer_call_and_return_conditional_losses_1813212
HidA3/StatefulPartitionedCall 
HidB/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0hidb_182734hidb_182736*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_HidB_layer_call_and_return_conditional_losses_1813382
HidB/StatefulPartitionedCall¥
HidB2/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0hidb2_182739hidb2_182741*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidB2_layer_call_and_return_conditional_losses_1813552
HidB2/StatefulPartitionedCall¥
HidB3/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0hidb3_182744hidb3_182746*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidB3_layer_call_and_return_conditional_losses_1813722
HidB3/StatefulPartitionedCall 
HidC/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0hidc_182749hidc_182751*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_HidC_layer_call_and_return_conditional_losses_1813892
HidC/StatefulPartitionedCall¥
HidC2/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0hidc2_182754hidc2_182756*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidC2_layer_call_and_return_conditional_losses_1814062
HidC2/StatefulPartitionedCall¥
HidD1/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0hidd1_182759hidd1_182761*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidD1_layer_call_and_return_conditional_losses_1814232
HidD1/StatefulPartitionedCall¥
HidE1/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0hide1_182764hide1_182766*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidE1_layer_call_and_return_conditional_losses_1814402
HidE1/StatefulPartitionedCallï
concatenate/PartitionedCallPartitionedCall%HidA/StatefulPartitionedCall:output:0&HidA2/StatefulPartitionedCall:output:0&HidA3/StatefulPartitionedCall:output:0%HidB/StatefulPartitionedCall:output:0&HidB2/StatefulPartitionedCall:output:0&HidB3/StatefulPartitionedCall:output:0%HidC/StatefulPartitionedCall:output:0&HidC2/StatefulPartitionedCall:output:0&HidD1/StatefulPartitionedCall:output:0&HidE1/StatefulPartitionedCall:output:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_1814612
concatenate/PartitionedCall¢
Hid1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0hid1_182770hid1_182772*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Hid1_layer_call_and_return_conditional_losses_1814742
Hid1/StatefulPartitionedCall£
Hid2/StatefulPartitionedCallStatefulPartitionedCall%Hid1/StatefulPartitionedCall:output:0hid2_182775hid2_182777*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Hid2_layer_call_and_return_conditional_losses_1814912
Hid2/StatefulPartitionedCall¬
output/StatefulPartitionedCallStatefulPartitionedCall%Hid2/StatefulPartitionedCall:output:0output_182780output_182782*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1815082 
output/StatefulPartitionedCall
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity¥
NoOpNoOp^ConA/StatefulPartitionedCall^ConA2/StatefulPartitionedCall^ConA3/StatefulPartitionedCall^ConB/StatefulPartitionedCall^ConB2/StatefulPartitionedCall^ConB3/StatefulPartitionedCall^ConC/StatefulPartitionedCall^ConC2/StatefulPartitionedCall^ConD/StatefulPartitionedCall^ConE/StatefulPartitionedCall^Hid1/StatefulPartitionedCall^Hid2/StatefulPartitionedCall^HidA/StatefulPartitionedCall^HidA2/StatefulPartitionedCall^HidA3/StatefulPartitionedCall^HidB/StatefulPartitionedCall^HidB2/StatefulPartitionedCall^HidB3/StatefulPartitionedCall^HidC/StatefulPartitionedCall^HidC2/StatefulPartitionedCall^HidD1/StatefulPartitionedCall^HidE1/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes{
y:ÿÿÿÿÿÿÿÿÿßÁ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
ConA/StatefulPartitionedCallConA/StatefulPartitionedCall2>
ConA2/StatefulPartitionedCallConA2/StatefulPartitionedCall2>
ConA3/StatefulPartitionedCallConA3/StatefulPartitionedCall2<
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
HidA3/StatefulPartitionedCallHidA3/StatefulPartitionedCall2<
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
:ÿÿÿÿÿÿÿÿÿßÁ

_user_specified_nameinput

ô
@__inference_Hid1_layer_call_and_return_conditional_losses_181474

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ö
A__inference_HidA2_layer_call_and_return_conditional_losses_181304

inputs3
matmul_readvariableop_resource:¤.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:¤*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¤: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
 
_user_specified_nameinputs
È
C
'__inference_PoolC2_layer_call_fn_184155

inputs
identityæ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolC2_layer_call_and_return_conditional_losses_1808322
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û
C
'__inference_PoolC2_layer_call_fn_184160

inputs
identityË
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_PoolC2_layer_call_and_return_conditional_losses_1811332
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿE<:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<
 
_user_specified_nameinputs

ô
@__inference_Hid1_layer_call_and_return_conditional_losses_184739

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
è
ú
A__inference_ConC2_layer_call_and_return_conditional_losses_181077

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿE<2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿG>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>
 
_user_specified_nameinputs
é
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_181250

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ Ô  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿH?:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH?
 
_user_specified_nameinputs
Æ
B
&__inference_PoolD_layer_call_fn_184175

inputs
identityå
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_PoolD_layer_call_and_return_conditional_losses_1808542
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø

&__inference_HidC2_layer_call_fn_184659

inputs
unknown:°
	unknown_0:	
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_HidC2_layer_call_and_return_conditional_losses_1814062
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ°: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
 
_user_specified_nameinputs
µ
^
B__inference_PoolB3_layer_call_and_return_conditional_losses_181139

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ!(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!(
 
_user_specified_nameinputs
Ä
c
*__inference_dropout_6_layer_call_fn_184254

inputs
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_1818582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
Ä
c
*__inference_dropout_2_layer_call_fn_184000

inputs
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_1820812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG>22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿG>
 
_user_specified_nameinputs"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¯
serving_default
A
input8
serving_default_input:0ÿÿÿÿÿÿÿÿÿßÁ:
output0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ(tensorflow/serving/predict:¢ë
æ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-5
layer-14
layer_with_weights-6
layer-15
layer_with_weights-7
layer-16
layer_with_weights-8
layer-17
layer_with_weights-9
layer-18
layer-19
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
*layer_with_weights-10
*layer-41
+layer_with_weights-11
+layer-42
,layer_with_weights-12
,layer-43
-layer_with_weights-13
-layer-44
.layer_with_weights-14
.layer-45
/layer_with_weights-15
/layer-46
0layer_with_weights-16
0layer-47
1layer_with_weights-17
1layer-48
2layer_with_weights-18
2layer-49
3layer_with_weights-19
3layer-50
4layer-51
5layer_with_weights-20
5layer-52
6layer_with_weights-21
6layer-53
7layer_with_weights-22
7layer-54
8	optimizer
9
signatures
#:_self_saveable_object_factories
;regularization_losses
<	variables
=trainable_variables
>	keras_api
+û&call_and_return_all_conditional_losses
ü_default_save_signature
ý__call__"
_tf_keras_network
D
#?_self_saveable_object_factories"
_tf_keras_input_layer
â

@kernel
Abias
#B_self_saveable_object_factories
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api
+þ&call_and_return_all_conditional_losses
ÿ__call__"
_tf_keras_layer
â

Gkernel
Hbias
#I_self_saveable_object_factories
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
Ì
#N_self_saveable_object_factories
Oregularization_losses
P	variables
Qtrainable_variables
R	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
Ì
#S_self_saveable_object_factories
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
Ì
#X_self_saveable_object_factories
Yregularization_losses
Z	variables
[trainable_variables
\	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
Ì
#]_self_saveable_object_factories
^regularization_losses
_	variables
`trainable_variables
a	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
â

bkernel
cbias
#d_self_saveable_object_factories
eregularization_losses
f	variables
gtrainable_variables
h	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
â

ikernel
jbias
#k_self_saveable_object_factories
lregularization_losses
m	variables
ntrainable_variables
o	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
â

pkernel
qbias
#r_self_saveable_object_factories
sregularization_losses
t	variables
utrainable_variables
v	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
Ì
#w_self_saveable_object_factories
xregularization_losses
y	variables
ztrainable_variables
{	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
Í
#|_self_saveable_object_factories
}regularization_losses
~	variables
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
Ñ
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
Ñ
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
é
kernel
	bias
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
é
kernel
	bias
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
é
kernel
	bias
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
é
 kernel
	¡bias
$¢_self_saveable_object_factories
£regularization_losses
¤	variables
¥trainable_variables
¦	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
é
§kernel
	¨bias
$©_self_saveable_object_factories
ªregularization_losses
«	variables
¬trainable_variables
­	keras_api
+ &call_and_return_all_conditional_losses
¡__call__"
_tf_keras_layer
Ñ
$®_self_saveable_object_factories
¯regularization_losses
°	variables
±trainable_variables
²	keras_api
+¢&call_and_return_all_conditional_losses
£__call__"
_tf_keras_layer
Ñ
$³_self_saveable_object_factories
´regularization_losses
µ	variables
¶trainable_variables
·	keras_api
+¤&call_and_return_all_conditional_losses
¥__call__"
_tf_keras_layer
Ñ
$¸_self_saveable_object_factories
¹regularization_losses
º	variables
»trainable_variables
¼	keras_api
+¦&call_and_return_all_conditional_losses
§__call__"
_tf_keras_layer
Ñ
$½_self_saveable_object_factories
¾regularization_losses
¿	variables
Àtrainable_variables
Á	keras_api
+¨&call_and_return_all_conditional_losses
©__call__"
_tf_keras_layer
Ñ
$Â_self_saveable_object_factories
Ãregularization_losses
Ä	variables
Åtrainable_variables
Æ	keras_api
+ª&call_and_return_all_conditional_losses
«__call__"
_tf_keras_layer
Ñ
$Ç_self_saveable_object_factories
Èregularization_losses
É	variables
Êtrainable_variables
Ë	keras_api
+¬&call_and_return_all_conditional_losses
­__call__"
_tf_keras_layer
Ñ
$Ì_self_saveable_object_factories
Íregularization_losses
Î	variables
Ïtrainable_variables
Ð	keras_api
+®&call_and_return_all_conditional_losses
¯__call__"
_tf_keras_layer
Ñ
$Ñ_self_saveable_object_factories
Òregularization_losses
Ó	variables
Ôtrainable_variables
Õ	keras_api
+°&call_and_return_all_conditional_losses
±__call__"
_tf_keras_layer
Ñ
$Ö_self_saveable_object_factories
×regularization_losses
Ø	variables
Ùtrainable_variables
Ú	keras_api
+²&call_and_return_all_conditional_losses
³__call__"
_tf_keras_layer
Ñ
$Û_self_saveable_object_factories
Üregularization_losses
Ý	variables
Þtrainable_variables
ß	keras_api
+´&call_and_return_all_conditional_losses
µ__call__"
_tf_keras_layer
Ñ
$à_self_saveable_object_factories
áregularization_losses
â	variables
ãtrainable_variables
ä	keras_api
+¶&call_and_return_all_conditional_losses
·__call__"
_tf_keras_layer
Ñ
$å_self_saveable_object_factories
æregularization_losses
ç	variables
ètrainable_variables
é	keras_api
+¸&call_and_return_all_conditional_losses
¹__call__"
_tf_keras_layer
Ñ
$ê_self_saveable_object_factories
ëregularization_losses
ì	variables
ítrainable_variables
î	keras_api
+º&call_and_return_all_conditional_losses
»__call__"
_tf_keras_layer
Ñ
$ï_self_saveable_object_factories
ðregularization_losses
ñ	variables
òtrainable_variables
ó	keras_api
+¼&call_and_return_all_conditional_losses
½__call__"
_tf_keras_layer
Ñ
$ô_self_saveable_object_factories
õregularization_losses
ö	variables
÷trainable_variables
ø	keras_api
+¾&call_and_return_all_conditional_losses
¿__call__"
_tf_keras_layer
Ñ
$ù_self_saveable_object_factories
úregularization_losses
û	variables
ütrainable_variables
ý	keras_api
+À&call_and_return_all_conditional_losses
Á__call__"
_tf_keras_layer
Ñ
$þ_self_saveable_object_factories
ÿregularization_losses
	variables
trainable_variables
	keras_api
+Â&call_and_return_all_conditional_losses
Ã__call__"
_tf_keras_layer
Ñ
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
+Ä&call_and_return_all_conditional_losses
Å__call__"
_tf_keras_layer
Ñ
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
+Æ&call_and_return_all_conditional_losses
Ç__call__"
_tf_keras_layer
Ñ
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
+È&call_and_return_all_conditional_losses
É__call__"
_tf_keras_layer
Ñ
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
+Ê&call_and_return_all_conditional_losses
Ë__call__"
_tf_keras_layer
Ñ
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
+Ì&call_and_return_all_conditional_losses
Í__call__"
_tf_keras_layer
é
kernel
	bias
$_self_saveable_object_factories
regularization_losses
 	variables
¡trainable_variables
¢	keras_api
+Î&call_and_return_all_conditional_losses
Ï__call__"
_tf_keras_layer
é
£kernel
	¤bias
$¥_self_saveable_object_factories
¦regularization_losses
§	variables
¨trainable_variables
©	keras_api
+Ð&call_and_return_all_conditional_losses
Ñ__call__"
_tf_keras_layer
é
ªkernel
	«bias
$¬_self_saveable_object_factories
­regularization_losses
®	variables
¯trainable_variables
°	keras_api
+Ò&call_and_return_all_conditional_losses
Ó__call__"
_tf_keras_layer
é
±kernel
	²bias
$³_self_saveable_object_factories
´regularization_losses
µ	variables
¶trainable_variables
·	keras_api
+Ô&call_and_return_all_conditional_losses
Õ__call__"
_tf_keras_layer
é
¸kernel
	¹bias
$º_self_saveable_object_factories
»regularization_losses
¼	variables
½trainable_variables
¾	keras_api
+Ö&call_and_return_all_conditional_losses
×__call__"
_tf_keras_layer
é
¿kernel
	Àbias
$Á_self_saveable_object_factories
Âregularization_losses
Ã	variables
Ätrainable_variables
Å	keras_api
+Ø&call_and_return_all_conditional_losses
Ù__call__"
_tf_keras_layer
é
Ækernel
	Çbias
$È_self_saveable_object_factories
Éregularization_losses
Ê	variables
Ëtrainable_variables
Ì	keras_api
+Ú&call_and_return_all_conditional_losses
Û__call__"
_tf_keras_layer
é
Íkernel
	Îbias
$Ï_self_saveable_object_factories
Ðregularization_losses
Ñ	variables
Òtrainable_variables
Ó	keras_api
+Ü&call_and_return_all_conditional_losses
Ý__call__"
_tf_keras_layer
é
Ôkernel
	Õbias
$Ö_self_saveable_object_factories
×regularization_losses
Ø	variables
Ùtrainable_variables
Ú	keras_api
+Þ&call_and_return_all_conditional_losses
ß__call__"
_tf_keras_layer
é
Ûkernel
	Übias
$Ý_self_saveable_object_factories
Þregularization_losses
ß	variables
àtrainable_variables
á	keras_api
+à&call_and_return_all_conditional_losses
á__call__"
_tf_keras_layer
Ñ
$â_self_saveable_object_factories
ãregularization_losses
ä	variables
åtrainable_variables
æ	keras_api
+â&call_and_return_all_conditional_losses
ã__call__"
_tf_keras_layer
é
çkernel
	èbias
$é_self_saveable_object_factories
êregularization_losses
ë	variables
ìtrainable_variables
í	keras_api
+ä&call_and_return_all_conditional_losses
å__call__"
_tf_keras_layer
é
îkernel
	ïbias
$ð_self_saveable_object_factories
ñregularization_losses
ò	variables
ótrainable_variables
ô	keras_api
+æ&call_and_return_all_conditional_losses
ç__call__"
_tf_keras_layer
é
õkernel
	öbias
$÷_self_saveable_object_factories
øregularization_losses
ù	variables
útrainable_variables
û	keras_api
+è&call_and_return_all_conditional_losses
é__call__"
_tf_keras_layer
¸
	üiter
ýbeta_1
þbeta_2

ÿdecay
learning_rate@mAm Gm¡Hm¢bm£cm¤im¥jm¦pm§qm¨	m©	mª	m«	m¬	m­	m®	 m¯	¡m°	§m±	¨m²	m³	m´	£mµ	¤m¶	ªm·	«m¸	±m¹	²mº	¸m»	¹m¼	¿m½	Àm¾	Æm¿	ÇmÀ	ÍmÁ	ÎmÂ	ÔmÃ	ÕmÄ	ÛmÅ	ÜmÆ	çmÇ	èmÈ	îmÉ	ïmÊ	õmË	ömÌ@vÍAvÎGvÏHvÐbvÑcvÒivÓjvÔpvÕqvÖ	v×	vØ	vÙ	vÚ	vÛ	vÜ	 vÝ	¡vÞ	§vß	¨và	vá	vâ	£vã	¤vä	ªvå	«væ	±vç	²vè	¸vé	¹vê	¿vë	Àvì	Æví	Çvî	Ívï	Îvð	Ôvñ	Õvò	Ûvó	Üvô	çvõ	èvö	îv÷	ïvø	õvù	övú"
	optimizer
-
êserving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
ª
@0
A1
G2
H3
b4
c5
i6
j7
p8
q9
10
11
12
13
14
15
 16
¡17
§18
¨19
20
21
£22
¤23
ª24
«25
±26
²27
¸28
¹29
¿30
À31
Æ32
Ç33
Í34
Î35
Ô36
Õ37
Û38
Ü39
ç40
è41
î42
ï43
õ44
ö45"
trackable_list_wrapper
ª
@0
A1
G2
H3
b4
c5
i6
j7
p8
q9
10
11
12
13
14
15
 16
¡17
§18
¨19
20
21
£22
¤23
ª24
«25
±26
²27
¸28
¹29
¿30
À31
Æ32
Ç33
Í34
Î35
Ô36
Õ37
Û38
Ü39
ç40
è41
î42
ï43
õ44
ö45"
trackable_list_wrapper
Ó
;regularization_losses
layers
 layer_regularization_losses
non_trainable_variables
<	variables
layer_metrics
metrics
=trainable_variables
ý__call__
ü_default_save_signature
+û&call_and_return_all_conditional_losses
'û"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
%:#2ConA/kernel
:2	ConA/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
µ
Cregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
D	variables
layer_metrics
metrics
Etrainable_variables
ÿ__call__
+þ&call_and_return_all_conditional_losses
'þ"call_and_return_conditional_losses"
_generic_user_object
%:#2ConB/kernel
:2	ConB/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
µ
Jregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
K	variables
layer_metrics
metrics
Ltrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Oregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
P	variables
layer_metrics
metrics
Qtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Tregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
U	variables
layer_metrics
metrics
Vtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Yregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
Z	variables
layer_metrics
metrics
[trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
^regularization_losses
layers
  layer_regularization_losses
¡non_trainable_variables
_	variables
¢layer_metrics
£metrics
`trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
%:#	2ConC/kernel
:2	ConC/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
µ
eregularization_losses
¤layers
 ¥layer_regularization_losses
¦non_trainable_variables
f	variables
§layer_metrics
¨metrics
gtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
&:$2ConA2/kernel
:2
ConA2/bias
 "
trackable_dict_wrapper
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
µ
lregularization_losses
©layers
 ªlayer_regularization_losses
«non_trainable_variables
m	variables
¬layer_metrics
­metrics
ntrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
&:$2ConB2/kernel
:2
ConB2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
µ
sregularization_losses
®layers
 ¯layer_regularization_losses
°non_trainable_variables
t	variables
±layer_metrics
²metrics
utrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
xregularization_losses
³layers
 ´layer_regularization_losses
µnon_trainable_variables
y	variables
¶layer_metrics
·metrics
ztrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
}regularization_losses
¸layers
 ¹layer_regularization_losses
ºnon_trainable_variables
~	variables
»layer_metrics
¼metrics
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
regularization_losses
½layers
 ¾layer_regularization_losses
¿non_trainable_variables
	variables
Àlayer_metrics
Ámetrics
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
regularization_losses
Âlayers
 Ãlayer_regularization_losses
Änon_trainable_variables
	variables
Ålayer_metrics
Æmetrics
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
&:$(2ConA3/kernel
:(2
ConA3/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
regularization_losses
Çlayers
 Èlayer_regularization_losses
Énon_trainable_variables
	variables
Êlayer_metrics
Ëmetrics
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
&:$(2ConB3/kernel
:(2
ConB3/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
regularization_losses
Ìlayers
 Ílayer_regularization_losses
Înon_trainable_variables
	variables
Ïlayer_metrics
Ðmetrics
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
&:$2ConC2/kernel
:2
ConC2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
regularization_losses
Ñlayers
 Òlayer_regularization_losses
Ónon_trainable_variables
	variables
Ôlayer_metrics
Õmetrics
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
%:#	2ConD/kernel
:2	ConD/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
 0
¡1"
trackable_list_wrapper
0
 0
¡1"
trackable_list_wrapper
¸
£regularization_losses
Ölayers
 ×layer_regularization_losses
Ønon_trainable_variables
¤	variables
Ùlayer_metrics
Úmetrics
¥trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
%:#2ConE/kernel
:2	ConE/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
§0
¨1"
trackable_list_wrapper
0
§0
¨1"
trackable_list_wrapper
¸
ªregularization_losses
Ûlayers
 Ülayer_regularization_losses
Ýnon_trainable_variables
«	variables
Þlayer_metrics
ßmetrics
¬trainable_variables
¡__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¯regularization_losses
àlayers
 álayer_regularization_losses
ânon_trainable_variables
°	variables
ãlayer_metrics
ämetrics
±trainable_variables
£__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
´regularization_losses
ålayers
 ælayer_regularization_losses
çnon_trainable_variables
µ	variables
èlayer_metrics
émetrics
¶trainable_variables
¥__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¹regularization_losses
êlayers
 ëlayer_regularization_losses
ìnon_trainable_variables
º	variables
ílayer_metrics
îmetrics
»trainable_variables
§__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¾regularization_losses
ïlayers
 ðlayer_regularization_losses
ñnon_trainable_variables
¿	variables
òlayer_metrics
ómetrics
Àtrainable_variables
©__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ãregularization_losses
ôlayers
 õlayer_regularization_losses
önon_trainable_variables
Ä	variables
÷layer_metrics
ømetrics
Åtrainable_variables
«__call__
+ª&call_and_return_all_conditional_losses
'ª"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Èregularization_losses
ùlayers
 úlayer_regularization_losses
ûnon_trainable_variables
É	variables
ülayer_metrics
ýmetrics
Êtrainable_variables
­__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Íregularization_losses
þlayers
 ÿlayer_regularization_losses
non_trainable_variables
Î	variables
layer_metrics
metrics
Ïtrainable_variables
¯__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Òregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
Ó	variables
layer_metrics
metrics
Ôtrainable_variables
±__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
×regularization_losses
layers
 layer_regularization_losses
non_trainable_variables
Ø	variables
layer_metrics
metrics
Ùtrainable_variables
³__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Üregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
Ý	variables
layer_metrics
metrics
Þtrainable_variables
µ__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
áregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
â	variables
layer_metrics
metrics
ãtrainable_variables
·__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
æregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
ç	variables
layer_metrics
metrics
ètrainable_variables
¹__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ëregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
ì	variables
layer_metrics
 metrics
ítrainable_variables
»__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ðregularization_losses
¡layers
 ¢layer_regularization_losses
£non_trainable_variables
ñ	variables
¤layer_metrics
¥metrics
òtrainable_variables
½__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
õregularization_losses
¦layers
 §layer_regularization_losses
¨non_trainable_variables
ö	variables
©layer_metrics
ªmetrics
÷trainable_variables
¿__call__
+¾&call_and_return_all_conditional_losses
'¾"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
úregularization_losses
«layers
 ¬layer_regularization_losses
­non_trainable_variables
û	variables
®layer_metrics
¯metrics
ütrainable_variables
Á__call__
+À&call_and_return_all_conditional_losses
'À"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ÿregularization_losses
°layers
 ±layer_regularization_losses
²non_trainable_variables
	variables
³layer_metrics
´metrics
trainable_variables
Ã__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
regularization_losses
µlayers
 ¶layer_regularization_losses
·non_trainable_variables
	variables
¸layer_metrics
¹metrics
trainable_variables
Å__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
regularization_losses
ºlayers
 »layer_regularization_losses
¼non_trainable_variables
	variables
½layer_metrics
¾metrics
trainable_variables
Ç__call__
+Æ&call_and_return_all_conditional_losses
'Æ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
regularization_losses
¿layers
 Àlayer_regularization_losses
Ánon_trainable_variables
	variables
Âlayer_metrics
Ãmetrics
trainable_variables
É__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
regularization_losses
Älayers
 Ålayer_regularization_losses
Ænon_trainable_variables
	variables
Çlayer_metrics
Èmetrics
trainable_variables
Ë__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
regularization_losses
Élayers
 Êlayer_regularization_losses
Ënon_trainable_variables
	variables
Ìlayer_metrics
Ímetrics
trainable_variables
Í__call__
+Ì&call_and_return_all_conditional_losses
'Ì"call_and_return_conditional_losses"
_generic_user_object
 :Ê×2HidA/kernel
:2	HidA/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
regularization_losses
Îlayers
 Ïlayer_regularization_losses
Ðnon_trainable_variables
 	variables
Ñlayer_metrics
Òmetrics
¡trainable_variables
Ï__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses"
_generic_user_object
!:¤2HidA2/kernel
:2
HidA2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
£0
¤1"
trackable_list_wrapper
0
£0
¤1"
trackable_list_wrapper
¸
¦regularization_losses
Ólayers
 Ôlayer_regularization_losses
Õnon_trainable_variables
§	variables
Ölayer_metrics
×metrics
¨trainable_variables
Ñ__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses"
_generic_user_object
 :
F2HidA3/kernel
:2
HidA3/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
ª0
«1"
trackable_list_wrapper
0
ª0
«1"
trackable_list_wrapper
¸
­regularization_losses
Ølayers
 Ùlayer_regularization_losses
Únon_trainable_variables
®	variables
Ûlayer_metrics
Ümetrics
¯trainable_variables
Ó__call__
+Ò&call_and_return_all_conditional_losses
'Ò"call_and_return_conditional_losses"
_generic_user_object
 : ©2HidB/kernel
:2	HidB/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
±0
²1"
trackable_list_wrapper
0
±0
²1"
trackable_list_wrapper
¸
´regularization_losses
Ýlayers
 Þlayer_regularization_losses
ßnon_trainable_variables
µ	variables
àlayer_metrics
ámetrics
¶trainable_variables
Õ__call__
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses"
_generic_user_object
!:¤2HidB2/kernel
:2
HidB2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
¸0
¹1"
trackable_list_wrapper
0
¸0
¹1"
trackable_list_wrapper
¸
»regularization_losses
âlayers
 ãlayer_regularization_losses
änon_trainable_variables
¼	variables
ålayer_metrics
æmetrics
½trainable_variables
×__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
 :
F2HidB3/kernel
:2
HidB3/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
¿0
À1"
trackable_list_wrapper
0
¿0
À1"
trackable_list_wrapper
¸
Âregularization_losses
çlayers
 èlayer_regularization_losses
énon_trainable_variables
Ã	variables
êlayer_metrics
ëmetrics
Ätrainable_variables
Ù__call__
+Ø&call_and_return_all_conditional_losses
'Ø"call_and_return_conditional_losses"
_generic_user_object
 :î2HidC/kernel
:2	HidC/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Æ0
Ç1"
trackable_list_wrapper
0
Æ0
Ç1"
trackable_list_wrapper
¸
Éregularization_losses
ìlayers
 ílayer_regularization_losses
înon_trainable_variables
Ê	variables
ïlayer_metrics
ðmetrics
Ëtrainable_variables
Û__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses"
_generic_user_object
!:°2HidC2/kernel
:2
HidC2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Í0
Î1"
trackable_list_wrapper
0
Í0
Î1"
trackable_list_wrapper
¸
Ðregularization_losses
ñlayers
 òlayer_regularization_losses
ónon_trainable_variables
Ñ	variables
ôlayer_metrics
õmetrics
Òtrainable_variables
Ý__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses"
_generic_user_object
!:á2HidD1/kernel
:2
HidD1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Ô0
Õ1"
trackable_list_wrapper
0
Ô0
Õ1"
trackable_list_wrapper
¸
×regularization_losses
ölayers
 ÷layer_regularization_losses
ønon_trainable_variables
Ø	variables
ùlayer_metrics
úmetrics
Ùtrainable_variables
ß__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses"
_generic_user_object
!: ð2HidE1/kernel
:2
HidE1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Û0
Ü1"
trackable_list_wrapper
0
Û0
Ü1"
trackable_list_wrapper
¸
Þregularization_losses
ûlayers
 ülayer_regularization_losses
ýnon_trainable_variables
ß	variables
þlayer_metrics
ÿmetrics
àtrainable_variables
á__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ãregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
ä	variables
layer_metrics
metrics
åtrainable_variables
ã__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses"
_generic_user_object
:
2Hid1/kernel
:2	Hid1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
ç0
è1"
trackable_list_wrapper
0
ç0
è1"
trackable_list_wrapper
¸
êregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
ë	variables
layer_metrics
metrics
ìtrainable_variables
å__call__
+ä&call_and_return_all_conditional_losses
'ä"call_and_return_conditional_losses"
_generic_user_object
:
2Hid2/kernel
:2	Hid2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
î0
ï1"
trackable_list_wrapper
0
î0
ï1"
trackable_list_wrapper
¸
ñregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
ò	variables
layer_metrics
metrics
ótrainable_variables
ç__call__
+æ&call_and_return_all_conditional_losses
'æ"call_and_return_conditional_losses"
_generic_user_object
 :	(2output/kernel
:(2output/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
õ0
ö1"
trackable_list_wrapper
0
õ0
ö1"
trackable_list_wrapper
¸
øregularization_losses
layers
 layer_regularization_losses
non_trainable_variables
ù	variables
layer_metrics
metrics
útrainable_variables
é__call__
+è&call_and_return_all_conditional_losses
'è"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
Î
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
754"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
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
R

total

count
	variables
	keras_api"
_tf_keras_metric
c

total

count

_fn_kwargs
	variables
	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
*:(2Adam/ConA/kernel/m
:2Adam/ConA/bias/m
*:(2Adam/ConB/kernel/m
:2Adam/ConB/bias/m
*:(	2Adam/ConC/kernel/m
:2Adam/ConC/bias/m
+:)2Adam/ConA2/kernel/m
:2Adam/ConA2/bias/m
+:)2Adam/ConB2/kernel/m
:2Adam/ConB2/bias/m
+:)(2Adam/ConA3/kernel/m
:(2Adam/ConA3/bias/m
+:)(2Adam/ConB3/kernel/m
:(2Adam/ConB3/bias/m
+:)2Adam/ConC2/kernel/m
:2Adam/ConC2/bias/m
*:(	2Adam/ConD/kernel/m
:2Adam/ConD/bias/m
*:(2Adam/ConE/kernel/m
:2Adam/ConE/bias/m
%:#Ê×2Adam/HidA/kernel/m
:2Adam/HidA/bias/m
&:$¤2Adam/HidA2/kernel/m
:2Adam/HidA2/bias/m
%:#
F2Adam/HidA3/kernel/m
:2Adam/HidA3/bias/m
%:# ©2Adam/HidB/kernel/m
:2Adam/HidB/bias/m
&:$¤2Adam/HidB2/kernel/m
:2Adam/HidB2/bias/m
%:#
F2Adam/HidB3/kernel/m
:2Adam/HidB3/bias/m
%:#î2Adam/HidC/kernel/m
:2Adam/HidC/bias/m
&:$°2Adam/HidC2/kernel/m
:2Adam/HidC2/bias/m
&:$á2Adam/HidD1/kernel/m
:2Adam/HidD1/bias/m
&:$ ð2Adam/HidE1/kernel/m
:2Adam/HidE1/bias/m
$:"
2Adam/Hid1/kernel/m
:2Adam/Hid1/bias/m
$:"
2Adam/Hid2/kernel/m
:2Adam/Hid2/bias/m
%:#	(2Adam/output/kernel/m
:(2Adam/output/bias/m
*:(2Adam/ConA/kernel/v
:2Adam/ConA/bias/v
*:(2Adam/ConB/kernel/v
:2Adam/ConB/bias/v
*:(	2Adam/ConC/kernel/v
:2Adam/ConC/bias/v
+:)2Adam/ConA2/kernel/v
:2Adam/ConA2/bias/v
+:)2Adam/ConB2/kernel/v
:2Adam/ConB2/bias/v
+:)(2Adam/ConA3/kernel/v
:(2Adam/ConA3/bias/v
+:)(2Adam/ConB3/kernel/v
:(2Adam/ConB3/bias/v
+:)2Adam/ConC2/kernel/v
:2Adam/ConC2/bias/v
*:(	2Adam/ConD/kernel/v
:2Adam/ConD/bias/v
*:(2Adam/ConE/kernel/v
:2Adam/ConE/bias/v
%:#Ê×2Adam/HidA/kernel/v
:2Adam/HidA/bias/v
&:$¤2Adam/HidA2/kernel/v
:2Adam/HidA2/bias/v
%:#
F2Adam/HidA3/kernel/v
:2Adam/HidA3/bias/v
%:# ©2Adam/HidB/kernel/v
:2Adam/HidB/bias/v
&:$¤2Adam/HidB2/kernel/v
:2Adam/HidB2/bias/v
%:#
F2Adam/HidB3/kernel/v
:2Adam/HidB3/bias/v
%:#î2Adam/HidC/kernel/v
:2Adam/HidC/bias/v
&:$°2Adam/HidC2/kernel/v
:2Adam/HidC2/bias/v
&:$á2Adam/HidD1/kernel/v
:2Adam/HidD1/bias/v
&:$ ð2Adam/HidE1/kernel/v
:2Adam/HidE1/bias/v
$:"
2Adam/Hid1/kernel/v
:2Adam/Hid1/bias/v
$:"
2Adam/Hid2/kernel/v
:2Adam/Hid2/bias/v
%:#	(2Adam/output/kernel/v
:(2Adam/output/bias/v
Ò2Ï
A__inference_model_layer_call_and_return_conditional_losses_183248
A__inference_model_layer_call_and_return_conditional_losses_183525
A__inference_model_layer_call_and_return_conditional_losses_182786
A__inference_model_layer_call_and_return_conditional_losses_182936À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÊBÇ
!__inference__wrapped_model_180669input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
æ2ã
&__inference_model_layer_call_fn_181610
&__inference_model_layer_call_fn_183622
&__inference_model_layer_call_fn_183719
&__inference_model_layer_call_fn_182636À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
@__inference_ConA_layer_call_and_return_conditional_losses_183730¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_ConA_layer_call_fn_183739¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_ConB_layer_call_and_return_conditional_losses_183750¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_ConB_layer_call_fn_183759¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
®2«
A__inference_PoolA_layer_call_and_return_conditional_losses_183764
A__inference_PoolA_layer_call_and_return_conditional_losses_183769¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
&__inference_PoolA_layer_call_fn_183774
&__inference_PoolA_layer_call_fn_183779¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
®2«
A__inference_PoolB_layer_call_and_return_conditional_losses_183784
A__inference_PoolB_layer_call_and_return_conditional_losses_183789¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
&__inference_PoolB_layer_call_fn_183794
&__inference_PoolB_layer_call_fn_183799¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ä2Á
C__inference_dropout_layer_call_and_return_conditional_losses_183804
C__inference_dropout_layer_call_and_return_conditional_losses_183816´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
(__inference_dropout_layer_call_fn_183821
(__inference_dropout_layer_call_fn_183826´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_1_layer_call_and_return_conditional_losses_183831
E__inference_dropout_1_layer_call_and_return_conditional_losses_183843´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
*__inference_dropout_1_layer_call_fn_183848
*__inference_dropout_1_layer_call_fn_183853´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
@__inference_ConC_layer_call_and_return_conditional_losses_183864¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_ConC_layer_call_fn_183873¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_ConA2_layer_call_and_return_conditional_losses_183884¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_ConA2_layer_call_fn_183893¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_ConB2_layer_call_and_return_conditional_losses_183904¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_ConB2_layer_call_fn_183913¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
®2«
A__inference_PoolC_layer_call_and_return_conditional_losses_183918
A__inference_PoolC_layer_call_and_return_conditional_losses_183923¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
&__inference_PoolC_layer_call_fn_183928
&__inference_PoolC_layer_call_fn_183933¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
°2­
B__inference_PoolA2_layer_call_and_return_conditional_losses_183938
B__inference_PoolA2_layer_call_and_return_conditional_losses_183943¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
'__inference_PoolA2_layer_call_fn_183948
'__inference_PoolA2_layer_call_fn_183953¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
°2­
B__inference_PoolB2_layer_call_and_return_conditional_losses_183958
B__inference_PoolB2_layer_call_and_return_conditional_losses_183963¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
'__inference_PoolB2_layer_call_fn_183968
'__inference_PoolB2_layer_call_fn_183973¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
È2Å
E__inference_dropout_2_layer_call_and_return_conditional_losses_183978
E__inference_dropout_2_layer_call_and_return_conditional_losses_183990´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
*__inference_dropout_2_layer_call_fn_183995
*__inference_dropout_2_layer_call_fn_184000´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ë2è
A__inference_ConA3_layer_call_and_return_conditional_losses_184011¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_ConA3_layer_call_fn_184020¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_ConB3_layer_call_and_return_conditional_losses_184031¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_ConB3_layer_call_fn_184040¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_ConC2_layer_call_and_return_conditional_losses_184051¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_ConC2_layer_call_fn_184060¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_ConD_layer_call_and_return_conditional_losses_184071¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_ConD_layer_call_fn_184080¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_ConE_layer_call_and_return_conditional_losses_184091¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_ConE_layer_call_fn_184100¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
°2­
B__inference_PoolA3_layer_call_and_return_conditional_losses_184105
B__inference_PoolA3_layer_call_and_return_conditional_losses_184110¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
'__inference_PoolA3_layer_call_fn_184115
'__inference_PoolA3_layer_call_fn_184120¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
°2­
B__inference_PoolB3_layer_call_and_return_conditional_losses_184125
B__inference_PoolB3_layer_call_and_return_conditional_losses_184130¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
'__inference_PoolB3_layer_call_fn_184135
'__inference_PoolB3_layer_call_fn_184140¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
°2­
B__inference_PoolC2_layer_call_and_return_conditional_losses_184145
B__inference_PoolC2_layer_call_and_return_conditional_losses_184150¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
'__inference_PoolC2_layer_call_fn_184155
'__inference_PoolC2_layer_call_fn_184160¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
®2«
A__inference_PoolD_layer_call_and_return_conditional_losses_184165
A__inference_PoolD_layer_call_and_return_conditional_losses_184170¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
&__inference_PoolD_layer_call_fn_184175
&__inference_PoolD_layer_call_fn_184180¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
®2«
A__inference_PoolE_layer_call_and_return_conditional_losses_184185
A__inference_PoolE_layer_call_and_return_conditional_losses_184190¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
&__inference_PoolE_layer_call_fn_184195
&__inference_PoolE_layer_call_fn_184200¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
È2Å
E__inference_dropout_5_layer_call_and_return_conditional_losses_184205
E__inference_dropout_5_layer_call_and_return_conditional_losses_184217´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
*__inference_dropout_5_layer_call_fn_184222
*__inference_dropout_5_layer_call_fn_184227´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_6_layer_call_and_return_conditional_losses_184232
E__inference_dropout_6_layer_call_and_return_conditional_losses_184244´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
*__inference_dropout_6_layer_call_fn_184249
*__inference_dropout_6_layer_call_fn_184254´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_7_layer_call_and_return_conditional_losses_184259
E__inference_dropout_7_layer_call_and_return_conditional_losses_184271´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
*__inference_dropout_7_layer_call_fn_184276
*__inference_dropout_7_layer_call_fn_184281´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_8_layer_call_and_return_conditional_losses_184286
E__inference_dropout_8_layer_call_and_return_conditional_losses_184298´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
*__inference_dropout_8_layer_call_fn_184303
*__inference_dropout_8_layer_call_fn_184308´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_9_layer_call_and_return_conditional_losses_184313
E__inference_dropout_9_layer_call_and_return_conditional_losses_184325´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
*__inference_dropout_9_layer_call_fn_184330
*__inference_dropout_9_layer_call_fn_184335´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_3_layer_call_and_return_conditional_losses_184340
E__inference_dropout_3_layer_call_and_return_conditional_losses_184352´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
*__inference_dropout_3_layer_call_fn_184357
*__inference_dropout_3_layer_call_fn_184362´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_4_layer_call_and_return_conditional_losses_184367
E__inference_dropout_4_layer_call_and_return_conditional_losses_184379´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
*__inference_dropout_4_layer_call_fn_184384
*__inference_dropout_4_layer_call_fn_184389´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
í2ê
C__inference_flatten_layer_call_and_return_conditional_losses_184395¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_flatten_layer_call_fn_184400¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_flatten_1_layer_call_and_return_conditional_losses_184406¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_flatten_1_layer_call_fn_184411¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_flatten_2_layer_call_and_return_conditional_losses_184417¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_flatten_2_layer_call_fn_184422¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_flatten_3_layer_call_and_return_conditional_losses_184428¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_flatten_3_layer_call_fn_184433¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_flatten_4_layer_call_and_return_conditional_losses_184439¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_flatten_4_layer_call_fn_184444¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_flatten_5_layer_call_and_return_conditional_losses_184450¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_flatten_5_layer_call_fn_184455¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_flatten_6_layer_call_and_return_conditional_losses_184461¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_flatten_6_layer_call_fn_184466¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_flatten_7_layer_call_and_return_conditional_losses_184472¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_flatten_7_layer_call_fn_184477¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_flatten_8_layer_call_and_return_conditional_losses_184483¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_flatten_8_layer_call_fn_184488¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_flatten_9_layer_call_and_return_conditional_losses_184494¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_flatten_9_layer_call_fn_184499¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_HidA_layer_call_and_return_conditional_losses_184510¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_HidA_layer_call_fn_184519¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_HidA2_layer_call_and_return_conditional_losses_184530¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_HidA2_layer_call_fn_184539¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_HidA3_layer_call_and_return_conditional_losses_184550¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_HidA3_layer_call_fn_184559¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_HidB_layer_call_and_return_conditional_losses_184570¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_HidB_layer_call_fn_184579¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_HidB2_layer_call_and_return_conditional_losses_184590¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_HidB2_layer_call_fn_184599¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_HidB3_layer_call_and_return_conditional_losses_184610¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_HidB3_layer_call_fn_184619¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_HidC_layer_call_and_return_conditional_losses_184630¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_HidC_layer_call_fn_184639¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_HidC2_layer_call_and_return_conditional_losses_184650¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_HidC2_layer_call_fn_184659¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_HidD1_layer_call_and_return_conditional_losses_184670¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_HidD1_layer_call_fn_184679¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_HidE1_layer_call_and_return_conditional_losses_184690¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_HidE1_layer_call_fn_184699¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_concatenate_layer_call_and_return_conditional_losses_184714¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_concatenate_layer_call_fn_184728¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_Hid1_layer_call_and_return_conditional_losses_184739¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_Hid1_layer_call_fn_184748¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_Hid2_layer_call_and_return_conditional_losses_184759¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_Hid2_layer_call_fn_184768¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_output_layer_call_and_return_conditional_losses_184779¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_output_layer_call_fn_184788¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÉBÆ
$__inference_signature_wrapper_183041input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ±
A__inference_ConA2_layer_call_and_return_conditional_losses_183884lij7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿI?
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿG=
 
&__inference_ConA2_layer_call_fn_183893_ij7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿI?
ª " ÿÿÿÿÿÿÿÿÿG=³
A__inference_ConA3_layer_call_and_return_conditional_losses_184011n7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ#
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ!(
 
&__inference_ConA3_layer_call_fn_184020a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ#
ª " ÿÿÿÿÿÿÿÿÿ!(´
@__inference_ConA_layer_call_and_return_conditional_losses_183730p@A9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿßÁ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿÜ¿
 
%__inference_ConA_layer_call_fn_183739c@A9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿßÁ
ª ""ÿÿÿÿÿÿÿÿÿÜ¿±
A__inference_ConB2_layer_call_and_return_conditional_losses_183904lpq7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿH?
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿF=
 
&__inference_ConB2_layer_call_fn_183913_pq7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿH?
ª " ÿÿÿÿÿÿÿÿÿF=³
A__inference_ConB3_layer_call_and_return_conditional_losses_184031n7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ#
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ!(
 
&__inference_ConB3_layer_call_fn_184040a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ#
ª " ÿÿÿÿÿÿÿÿÿ!(´
@__inference_ConB_layer_call_and_return_conditional_losses_183750pGH9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿßÁ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿÙ½
 
%__inference_ConB_layer_call_fn_183759cGH9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿßÁ
ª ""ÿÿÿÿÿÿÿÿÿÙ½³
A__inference_ConC2_layer_call_and_return_conditional_losses_184051n7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿG>
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿE<
 
&__inference_ConC2_layer_call_fn_184060a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿG>
ª " ÿÿÿÿÿÿÿÿÿE<´
@__inference_ConC_layer_call_and_return_conditional_losses_183864pbc9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿßÁ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ×»
 
%__inference_ConC_layer_call_fn_183873cbc9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿßÁ
ª ""ÿÿÿÿÿÿÿÿÿ×»¶
@__inference_ConD_layer_call_and_return_conditional_losses_184071r ¡9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿßÁ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿÕ¹
 
%__inference_ConD_layer_call_fn_184080e ¡9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿßÁ
ª ""ÿÿÿÿÿÿÿÿÿÕ¹¶
@__inference_ConE_layer_call_and_return_conditional_losses_184091r§¨9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿßÁ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿÍ³
 
%__inference_ConE_layer_call_fn_184100e§¨9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿßÁ
ª ""ÿÿÿÿÿÿÿÿÿÍ³¤
@__inference_Hid1_layer_call_and_return_conditional_losses_184739`çè0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 |
%__inference_Hid1_layer_call_fn_184748Sçè0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
@__inference_Hid2_layer_call_and_return_conditional_losses_184759`îï0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 |
%__inference_Hid2_layer_call_fn_184768Sîï0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
A__inference_HidA2_layer_call_and_return_conditional_losses_184530a£¤1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ¤
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
&__inference_HidA2_layer_call_fn_184539T£¤1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ¤
ª "ÿÿÿÿÿÿÿÿÿ¥
A__inference_HidA3_layer_call_and_return_conditional_losses_184550`ª«0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿF
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
&__inference_HidA3_layer_call_fn_184559Sª«0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿF
ª "ÿÿÿÿÿÿÿÿÿ¥
@__inference_HidA_layer_call_and_return_conditional_losses_184510a1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿÊ×
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
%__inference_HidA_layer_call_fn_184519T1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿÊ×
ª "ÿÿÿÿÿÿÿÿÿ¦
A__inference_HidB2_layer_call_and_return_conditional_losses_184590a¸¹1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ¤
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
&__inference_HidB2_layer_call_fn_184599T¸¹1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ¤
ª "ÿÿÿÿÿÿÿÿÿ¥
A__inference_HidB3_layer_call_and_return_conditional_losses_184610`¿À0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿF
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
&__inference_HidB3_layer_call_fn_184619S¿À0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿF
ª "ÿÿÿÿÿÿÿÿÿ¥
@__inference_HidB_layer_call_and_return_conditional_losses_184570a±²1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ ©
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
%__inference_HidB_layer_call_fn_184579T±²1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ ©
ª "ÿÿÿÿÿÿÿÿÿ¦
A__inference_HidC2_layer_call_and_return_conditional_losses_184650aÍÎ1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ°
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
&__inference_HidC2_layer_call_fn_184659TÍÎ1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ°
ª "ÿÿÿÿÿÿÿÿÿ¥
@__inference_HidC_layer_call_and_return_conditional_losses_184630aÆÇ1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿî
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
%__inference_HidC_layer_call_fn_184639TÆÇ1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿî
ª "ÿÿÿÿÿÿÿÿÿ¦
A__inference_HidD1_layer_call_and_return_conditional_losses_184670aÔÕ1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿá
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
&__inference_HidD1_layer_call_fn_184679TÔÕ1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿá
ª "ÿÿÿÿÿÿÿÿÿ¦
A__inference_HidE1_layer_call_and_return_conditional_losses_184690aÛÜ1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ ð
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
&__inference_HidE1_layer_call_fn_184699TÛÜ1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ ð
ª "ÿÿÿÿÿÿÿÿÿå
B__inference_PoolA2_layer_call_and_return_conditional_losses_183938R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ®
B__inference_PoolA2_layer_call_and_return_conditional_losses_183943h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿG=
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ#
 ½
'__inference_PoolA2_layer_call_fn_183948R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
'__inference_PoolA2_layer_call_fn_183953[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿG=
ª " ÿÿÿÿÿÿÿÿÿ#å
B__inference_PoolA3_layer_call_and_return_conditional_losses_184105R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ®
B__inference_PoolA3_layer_call_and_return_conditional_losses_184110h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ!(
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ(
 ½
'__inference_PoolA3_layer_call_fn_184115R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
'__inference_PoolA3_layer_call_fn_184120[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ!(
ª " ÿÿÿÿÿÿÿÿÿ(ä
A__inference_PoolA_layer_call_and_return_conditional_losses_183764R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¯
A__inference_PoolA_layer_call_and_return_conditional_losses_183769j9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿÜ¿
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿI?
 ¼
&__inference_PoolA_layer_call_fn_183774R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
&__inference_PoolA_layer_call_fn_183779]9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿÜ¿
ª " ÿÿÿÿÿÿÿÿÿI?å
B__inference_PoolB2_layer_call_and_return_conditional_losses_183958R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ®
B__inference_PoolB2_layer_call_and_return_conditional_losses_183963h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿF=
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ#
 ½
'__inference_PoolB2_layer_call_fn_183968R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
'__inference_PoolB2_layer_call_fn_183973[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿF=
ª " ÿÿÿÿÿÿÿÿÿ#å
B__inference_PoolB3_layer_call_and_return_conditional_losses_184125R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ®
B__inference_PoolB3_layer_call_and_return_conditional_losses_184130h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ!(
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ(
 ½
'__inference_PoolB3_layer_call_fn_184135R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
'__inference_PoolB3_layer_call_fn_184140[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ!(
ª " ÿÿÿÿÿÿÿÿÿ(ä
A__inference_PoolB_layer_call_and_return_conditional_losses_183784R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¯
A__inference_PoolB_layer_call_and_return_conditional_losses_183789j9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿÙ½
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿH?
 ¼
&__inference_PoolB_layer_call_fn_183794R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
&__inference_PoolB_layer_call_fn_183799]9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿÙ½
ª " ÿÿÿÿÿÿÿÿÿH?å
B__inference_PoolC2_layer_call_and_return_conditional_losses_184145R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ®
B__inference_PoolC2_layer_call_and_return_conditional_losses_184150h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿE<
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ"
 ½
'__inference_PoolC2_layer_call_fn_184155R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
'__inference_PoolC2_layer_call_fn_184160[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿE<
ª " ÿÿÿÿÿÿÿÿÿ"ä
A__inference_PoolC_layer_call_and_return_conditional_losses_183918R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¯
A__inference_PoolC_layer_call_and_return_conditional_losses_183923j9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ×»
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿG>
 ¼
&__inference_PoolC_layer_call_fn_183928R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
&__inference_PoolC_layer_call_fn_183933]9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ×»
ª " ÿÿÿÿÿÿÿÿÿG>ä
A__inference_PoolD_layer_call_and_return_conditional_losses_184165R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¯
A__inference_PoolD_layer_call_and_return_conditional_losses_184170j9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿÕ¹
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿG=
 ¼
&__inference_PoolD_layer_call_fn_184175R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
&__inference_PoolD_layer_call_fn_184180]9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿÕ¹
ª " ÿÿÿÿÿÿÿÿÿG=ä
A__inference_PoolE_layer_call_and_return_conditional_losses_184185R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¯
A__inference_PoolE_layer_call_and_return_conditional_losses_184190j9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿÍ³
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿD;
 ¼
&__inference_PoolE_layer_call_fn_184195R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
&__inference_PoolE_layer_call_fn_184200]9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿÍ³
ª " ÿÿÿÿÿÿÿÿÿD;å
!__inference__wrapped_model_180669¿RGH@Abcpqij§¨ ¡£¤ª«±²¸¹¿ÀÆÇÍÎÔÕÛÜçèîïõö8¢5
.¢+
)&
inputÿÿÿÿÿÿÿÿÿßÁ
ª "/ª,
*
output 
outputÿÿÿÿÿÿÿÿÿ(
G__inference_concatenate_layer_call_and_return_conditional_losses_184714´¢
ý¢ù
öò
# 
inputs/0ÿÿÿÿÿÿÿÿÿ
# 
inputs/1ÿÿÿÿÿÿÿÿÿ
# 
inputs/2ÿÿÿÿÿÿÿÿÿ
# 
inputs/3ÿÿÿÿÿÿÿÿÿ
# 
inputs/4ÿÿÿÿÿÿÿÿÿ
# 
inputs/5ÿÿÿÿÿÿÿÿÿ
# 
inputs/6ÿÿÿÿÿÿÿÿÿ
# 
inputs/7ÿÿÿÿÿÿÿÿÿ
# 
inputs/8ÿÿÿÿÿÿÿÿÿ
# 
inputs/9ÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 Ø
,__inference_concatenate_layer_call_fn_184728§¢
ý¢ù
öò
# 
inputs/0ÿÿÿÿÿÿÿÿÿ
# 
inputs/1ÿÿÿÿÿÿÿÿÿ
# 
inputs/2ÿÿÿÿÿÿÿÿÿ
# 
inputs/3ÿÿÿÿÿÿÿÿÿ
# 
inputs/4ÿÿÿÿÿÿÿÿÿ
# 
inputs/5ÿÿÿÿÿÿÿÿÿ
# 
inputs/6ÿÿÿÿÿÿÿÿÿ
# 
inputs/7ÿÿÿÿÿÿÿÿÿ
# 
inputs/8ÿÿÿÿÿÿÿÿÿ
# 
inputs/9ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿµ
E__inference_dropout_1_layer_call_and_return_conditional_losses_183831l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿH?
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿH?
 µ
E__inference_dropout_1_layer_call_and_return_conditional_losses_183843l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿH?
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿH?
 
*__inference_dropout_1_layer_call_fn_183848_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿH?
p 
ª " ÿÿÿÿÿÿÿÿÿH?
*__inference_dropout_1_layer_call_fn_183853_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿH?
p
ª " ÿÿÿÿÿÿÿÿÿH?µ
E__inference_dropout_2_layer_call_and_return_conditional_losses_183978l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿG>
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿG>
 µ
E__inference_dropout_2_layer_call_and_return_conditional_losses_183990l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿG>
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿG>
 
*__inference_dropout_2_layer_call_fn_183995_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿG>
p 
ª " ÿÿÿÿÿÿÿÿÿG>
*__inference_dropout_2_layer_call_fn_184000_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿG>
p
ª " ÿÿÿÿÿÿÿÿÿG>µ
E__inference_dropout_3_layer_call_and_return_conditional_losses_184340l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿG=
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿG=
 µ
E__inference_dropout_3_layer_call_and_return_conditional_losses_184352l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿG=
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿG=
 
*__inference_dropout_3_layer_call_fn_184357_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿG=
p 
ª " ÿÿÿÿÿÿÿÿÿG=
*__inference_dropout_3_layer_call_fn_184362_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿG=
p
ª " ÿÿÿÿÿÿÿÿÿG=µ
E__inference_dropout_4_layer_call_and_return_conditional_losses_184367l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿD;
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿD;
 µ
E__inference_dropout_4_layer_call_and_return_conditional_losses_184379l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿD;
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿD;
 
*__inference_dropout_4_layer_call_fn_184384_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿD;
p 
ª " ÿÿÿÿÿÿÿÿÿD;
*__inference_dropout_4_layer_call_fn_184389_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿD;
p
ª " ÿÿÿÿÿÿÿÿÿD;µ
E__inference_dropout_5_layer_call_and_return_conditional_losses_184205l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ#
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ#
 µ
E__inference_dropout_5_layer_call_and_return_conditional_losses_184217l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ#
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ#
 
*__inference_dropout_5_layer_call_fn_184222_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ#
p 
ª " ÿÿÿÿÿÿÿÿÿ#
*__inference_dropout_5_layer_call_fn_184227_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ#
p
ª " ÿÿÿÿÿÿÿÿÿ#µ
E__inference_dropout_6_layer_call_and_return_conditional_losses_184232l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ(
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ(
 µ
E__inference_dropout_6_layer_call_and_return_conditional_losses_184244l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ(
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ(
 
*__inference_dropout_6_layer_call_fn_184249_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ(
p 
ª " ÿÿÿÿÿÿÿÿÿ(
*__inference_dropout_6_layer_call_fn_184254_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ(
p
ª " ÿÿÿÿÿÿÿÿÿ(µ
E__inference_dropout_7_layer_call_and_return_conditional_losses_184259l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ#
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ#
 µ
E__inference_dropout_7_layer_call_and_return_conditional_losses_184271l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ#
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ#
 
*__inference_dropout_7_layer_call_fn_184276_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ#
p 
ª " ÿÿÿÿÿÿÿÿÿ#
*__inference_dropout_7_layer_call_fn_184281_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ#
p
ª " ÿÿÿÿÿÿÿÿÿ#µ
E__inference_dropout_8_layer_call_and_return_conditional_losses_184286l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ(
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ(
 µ
E__inference_dropout_8_layer_call_and_return_conditional_losses_184298l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ(
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ(
 
*__inference_dropout_8_layer_call_fn_184303_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ(
p 
ª " ÿÿÿÿÿÿÿÿÿ(
*__inference_dropout_8_layer_call_fn_184308_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ(
p
ª " ÿÿÿÿÿÿÿÿÿ(µ
E__inference_dropout_9_layer_call_and_return_conditional_losses_184313l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ"
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ"
 µ
E__inference_dropout_9_layer_call_and_return_conditional_losses_184325l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ"
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ"
 
*__inference_dropout_9_layer_call_fn_184330_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ"
p 
ª " ÿÿÿÿÿÿÿÿÿ"
*__inference_dropout_9_layer_call_fn_184335_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ"
p
ª " ÿÿÿÿÿÿÿÿÿ"³
C__inference_dropout_layer_call_and_return_conditional_losses_183804l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿI?
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿI?
 ³
C__inference_dropout_layer_call_and_return_conditional_losses_183816l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿI?
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿI?
 
(__inference_dropout_layer_call_fn_183821_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿI?
p 
ª " ÿÿÿÿÿÿÿÿÿI?
(__inference_dropout_layer_call_fn_183826_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿI?
p
ª " ÿÿÿÿÿÿÿÿÿI?«
E__inference_flatten_1_layer_call_and_return_conditional_losses_184406b7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ#
ª "'¢$

0ÿÿÿÿÿÿÿÿÿ¤
 
*__inference_flatten_1_layer_call_fn_184411U7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ#
ª "ÿÿÿÿÿÿÿÿÿ¤ª
E__inference_flatten_2_layer_call_and_return_conditional_losses_184417a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ(
ª "&¢#

0ÿÿÿÿÿÿÿÿÿF
 
*__inference_flatten_2_layer_call_fn_184422T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ(
ª "ÿÿÿÿÿÿÿÿÿF«
E__inference_flatten_3_layer_call_and_return_conditional_losses_184428b7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿH?
ª "'¢$

0ÿÿÿÿÿÿÿÿÿ ©
 
*__inference_flatten_3_layer_call_fn_184433U7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿH?
ª "ÿÿÿÿÿÿÿÿÿ ©«
E__inference_flatten_4_layer_call_and_return_conditional_losses_184439b7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ#
ª "'¢$

0ÿÿÿÿÿÿÿÿÿ¤
 
*__inference_flatten_4_layer_call_fn_184444U7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ#
ª "ÿÿÿÿÿÿÿÿÿ¤ª
E__inference_flatten_5_layer_call_and_return_conditional_losses_184450a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ(
ª "&¢#

0ÿÿÿÿÿÿÿÿÿF
 
*__inference_flatten_5_layer_call_fn_184455T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ(
ª "ÿÿÿÿÿÿÿÿÿF«
E__inference_flatten_6_layer_call_and_return_conditional_losses_184461b7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿG>
ª "'¢$

0ÿÿÿÿÿÿÿÿÿî
 
*__inference_flatten_6_layer_call_fn_184466U7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿG>
ª "ÿÿÿÿÿÿÿÿÿî«
E__inference_flatten_7_layer_call_and_return_conditional_losses_184472b7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ"
ª "'¢$

0ÿÿÿÿÿÿÿÿÿ°
 
*__inference_flatten_7_layer_call_fn_184477U7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ"
ª "ÿÿÿÿÿÿÿÿÿ°«
E__inference_flatten_8_layer_call_and_return_conditional_losses_184483b7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿG=
ª "'¢$

0ÿÿÿÿÿÿÿÿÿá
 
*__inference_flatten_8_layer_call_fn_184488U7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿG=
ª "ÿÿÿÿÿÿÿÿÿá«
E__inference_flatten_9_layer_call_and_return_conditional_losses_184494b7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿD;
ª "'¢$

0ÿÿÿÿÿÿÿÿÿ ð
 
*__inference_flatten_9_layer_call_fn_184499U7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿD;
ª "ÿÿÿÿÿÿÿÿÿ ð©
C__inference_flatten_layer_call_and_return_conditional_losses_184395b7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿI?
ª "'¢$

0ÿÿÿÿÿÿÿÿÿÊ×
 
(__inference_flatten_layer_call_fn_184400U7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿI?
ª "ÿÿÿÿÿÿÿÿÿÊ×
A__inference_model_layer_call_and_return_conditional_losses_182786½RGH@Abcpqij§¨ ¡£¤ª«±²¸¹¿ÀÆÇÍÎÔÕÛÜçèîïõö@¢=
6¢3
)&
inputÿÿÿÿÿÿÿÿÿßÁ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ(
 
A__inference_model_layer_call_and_return_conditional_losses_182936½RGH@Abcpqij§¨ ¡£¤ª«±²¸¹¿ÀÆÇÍÎÔÕÛÜçèîïõö@¢=
6¢3
)&
inputÿÿÿÿÿÿÿÿÿßÁ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ(
 
A__inference_model_layer_call_and_return_conditional_losses_183248¾RGH@Abcpqij§¨ ¡£¤ª«±²¸¹¿ÀÆÇÍÎÔÕÛÜçèîïõöA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿßÁ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ(
 
A__inference_model_layer_call_and_return_conditional_losses_183525¾RGH@Abcpqij§¨ ¡£¤ª«±²¸¹¿ÀÆÇÍÎÔÕÛÜçèîïõöA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿßÁ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ(
 Û
&__inference_model_layer_call_fn_181610°RGH@Abcpqij§¨ ¡£¤ª«±²¸¹¿ÀÆÇÍÎÔÕÛÜçèîïõö@¢=
6¢3
)&
inputÿÿÿÿÿÿÿÿÿßÁ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ(Û
&__inference_model_layer_call_fn_182636°RGH@Abcpqij§¨ ¡£¤ª«±²¸¹¿ÀÆÇÍÎÔÕÛÜçèîïõö@¢=
6¢3
)&
inputÿÿÿÿÿÿÿÿÿßÁ
p

 
ª "ÿÿÿÿÿÿÿÿÿ(Ü
&__inference_model_layer_call_fn_183622±RGH@Abcpqij§¨ ¡£¤ª«±²¸¹¿ÀÆÇÍÎÔÕÛÜçèîïõöA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿßÁ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ(Ü
&__inference_model_layer_call_fn_183719±RGH@Abcpqij§¨ ¡£¤ª«±²¸¹¿ÀÆÇÍÎÔÕÛÜçèîïõöA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿßÁ
p

 
ª "ÿÿÿÿÿÿÿÿÿ(¥
B__inference_output_layer_call_and_return_conditional_losses_184779_õö0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ(
 }
'__inference_output_layer_call_fn_184788Rõö0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ(ñ
$__inference_signature_wrapper_183041ÈRGH@Abcpqij§¨ ¡£¤ª«±²¸¹¿ÀÆÇÍÎÔÕÛÜçèîïõöA¢>
¢ 
7ª4
2
input)&
inputÿÿÿÿÿÿÿÿÿßÁ"/ª,
*
output 
outputÿÿÿÿÿÿÿÿÿ(