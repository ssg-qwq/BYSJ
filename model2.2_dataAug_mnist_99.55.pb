??+
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
 ?"serve*2.6.22v2.6.1-9-gc2363d6d0258??%
z
ConA/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConA/kernel
s
ConA/kernel/Read/ReadVariableOpReadVariableOpConA/kernel*&
_output_shapes
:*
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
shape:*
shared_nameConB/kernel
s
ConB/kernel/Read/ReadVariableOpReadVariableOpConB/kernel*&
_output_shapes
:*
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
z
ConC/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConC/kernel
s
ConC/kernel/Read/ReadVariableOpReadVariableOpConC/kernel*&
_output_shapes
:*
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
z
ConD/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*
shared_nameConD/kernel
s
ConD/kernel/Read/ReadVariableOpReadVariableOpConD/kernel*&
_output_shapes
:		*
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
shape:*
shared_nameConE/kernel
s
ConE/kernel/Read/ReadVariableOpReadVariableOpConE/kernel*&
_output_shapes
:*
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
t
HidA/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameHidA/kernel
m
HidA/kernel/Read/ReadVariableOpReadVariableOpHidA/kernel* 
_output_shapes
:
??*
dtype0
k
	HidA/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	HidA/bias
d
HidA/bias/Read/ReadVariableOpReadVariableOp	HidA/bias*
_output_shapes	
:?*
dtype0
v
HidA2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameHidA2/kernel
o
 HidA2/kernel/Read/ReadVariableOpReadVariableOpHidA2/kernel* 
_output_shapes
:
??*
dtype0
m

HidA2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
HidA2/bias
f
HidA2/bias/Read/ReadVariableOpReadVariableOp
HidA2/bias*
_output_shapes	
:?*
dtype0
u
HidA3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(?*
shared_nameHidA3/kernel
n
 HidA3/kernel/Read/ReadVariableOpReadVariableOpHidA3/kernel*
_output_shapes
:	(?*
dtype0
m

HidA3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
HidA3/bias
f
HidA3/bias/Read/ReadVariableOpReadVariableOp
HidA3/bias*
_output_shapes	
:?*
dtype0
t
HidB/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameHidB/kernel
m
HidB/kernel/Read/ReadVariableOpReadVariableOpHidB/kernel* 
_output_shapes
:
??*
dtype0
k
	HidB/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	HidB/bias
d
HidB/bias/Read/ReadVariableOpReadVariableOp	HidB/bias*
_output_shapes	
:?*
dtype0
v
HidB2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameHidB2/kernel
o
 HidB2/kernel/Read/ReadVariableOpReadVariableOpHidB2/kernel* 
_output_shapes
:
??*
dtype0
m

HidB2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
HidB2/bias
f
HidB2/bias/Read/ReadVariableOpReadVariableOp
HidB2/bias*
_output_shapes	
:?*
dtype0
u
HidB3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(?*
shared_nameHidB3/kernel
n
 HidB3/kernel/Read/ReadVariableOpReadVariableOpHidB3/kernel*
_output_shapes
:	(?*
dtype0
m

HidB3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
HidB3/bias
f
HidB3/bias/Read/ReadVariableOpReadVariableOp
HidB3/bias*
_output_shapes	
:?*
dtype0
t
HidC/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameHidC/kernel
m
HidC/kernel/Read/ReadVariableOpReadVariableOpHidC/kernel* 
_output_shapes
:
??*
dtype0
k
	HidC/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	HidC/bias
d
HidC/bias/Read/ReadVariableOpReadVariableOp	HidC/bias*
_output_shapes	
:?*
dtype0
v
HidD1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameHidD1/kernel
o
 HidD1/kernel/Read/ReadVariableOpReadVariableOpHidD1/kernel* 
_output_shapes
:
??*
dtype0
m

HidD1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
HidD1/bias
f
HidD1/bias/Read/ReadVariableOpReadVariableOp
HidD1/bias*
_output_shapes	
:?*
dtype0
v
HidE1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameHidE1/kernel
o
 HidE1/kernel/Read/ReadVariableOpReadVariableOpHidE1/kernel* 
_output_shapes
:
??*
dtype0
m

HidE1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
HidE1/bias
f
HidE1/bias/Read/ReadVariableOpReadVariableOp
HidE1/bias*
_output_shapes	
:?*
dtype0
t
Hid1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?$?*
shared_nameHid1/kernel
m
Hid1/kernel/Read/ReadVariableOpReadVariableOpHid1/kernel* 
_output_shapes
:
?$?*
dtype0
k
	Hid1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	Hid1/bias
d
Hid1/bias/Read/ReadVariableOpReadVariableOp	Hid1/bias*
_output_shapes	
:?*
dtype0
t
Hid2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameHid2/kernel
m
Hid2/kernel/Read/ReadVariableOpReadVariableOpHid2/kernel* 
_output_shapes
:
??*
dtype0
k
	Hid2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	Hid2/bias
d
Hid2/bias/Read/ReadVariableOpReadVariableOp	Hid2/bias*
_output_shapes	
:?*
dtype0
t
Hid3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameHid3/kernel
m
Hid3/kernel/Read/ReadVariableOpReadVariableOpHid3/kernel* 
_output_shapes
:
??*
dtype0
k
	Hid3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	Hid3/bias
d
Hid3/bias/Read/ReadVariableOpReadVariableOp	Hid3/bias*
_output_shapes	
:?*
dtype0
w
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*
shared_nameoutput/kernel
p
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes
:	?
*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:
*
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
shape:*#
shared_nameAdam/ConA/kernel/m
?
&Adam/ConA/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConA/kernel/m*&
_output_shapes
:*
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
?
Adam/ConB/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/ConB/kernel/m
?
&Adam/ConB/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConB/kernel/m*&
_output_shapes
:*
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
?
Adam/ConA2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConA2/kernel/m
?
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
?
Adam/ConB2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConB2/kernel/m
?
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
Adam/ConC/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/ConC/kernel/m
?
&Adam/ConC/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConC/kernel/m*&
_output_shapes
:*
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
?
Adam/ConD/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*#
shared_nameAdam/ConD/kernel/m
?
&Adam/ConD/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConD/kernel/m*&
_output_shapes
:		*
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
?
Adam/ConE/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/ConE/kernel/m
?
&Adam/ConE/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ConE/kernel/m*&
_output_shapes
:*
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
?
Adam/HidA/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_nameAdam/HidA/kernel/m
{
&Adam/HidA/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidA/kernel/m* 
_output_shapes
:
??*
dtype0
y
Adam/HidA/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/HidA/bias/m
r
$Adam/HidA/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidA/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/HidA2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/HidA2/kernel/m
}
'Adam/HidA2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidA2/kernel/m* 
_output_shapes
:
??*
dtype0
{
Adam/HidA2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidA2/bias/m
t
%Adam/HidA2/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidA2/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/HidA3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(?*$
shared_nameAdam/HidA3/kernel/m
|
'Adam/HidA3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidA3/kernel/m*
_output_shapes
:	(?*
dtype0
{
Adam/HidA3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidA3/bias/m
t
%Adam/HidA3/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidA3/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/HidB/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_nameAdam/HidB/kernel/m
{
&Adam/HidB/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidB/kernel/m* 
_output_shapes
:
??*
dtype0
y
Adam/HidB/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/HidB/bias/m
r
$Adam/HidB/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidB/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/HidB2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/HidB2/kernel/m
}
'Adam/HidB2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidB2/kernel/m* 
_output_shapes
:
??*
dtype0
{
Adam/HidB2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidB2/bias/m
t
%Adam/HidB2/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidB2/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/HidB3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(?*$
shared_nameAdam/HidB3/kernel/m
|
'Adam/HidB3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidB3/kernel/m*
_output_shapes
:	(?*
dtype0
{
Adam/HidB3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidB3/bias/m
t
%Adam/HidB3/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidB3/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/HidC/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_nameAdam/HidC/kernel/m
{
&Adam/HidC/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidC/kernel/m* 
_output_shapes
:
??*
dtype0
y
Adam/HidC/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/HidC/bias/m
r
$Adam/HidC/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidC/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/HidD1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/HidD1/kernel/m
}
'Adam/HidD1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidD1/kernel/m* 
_output_shapes
:
??*
dtype0
{
Adam/HidD1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidD1/bias/m
t
%Adam/HidD1/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidD1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/HidE1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/HidE1/kernel/m
}
'Adam/HidE1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HidE1/kernel/m* 
_output_shapes
:
??*
dtype0
{
Adam/HidE1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidE1/bias/m
t
%Adam/HidE1/bias/m/Read/ReadVariableOpReadVariableOpAdam/HidE1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/Hid1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?$?*#
shared_nameAdam/Hid1/kernel/m
{
&Adam/Hid1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Hid1/kernel/m* 
_output_shapes
:
?$?*
dtype0
y
Adam/Hid1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/Hid1/bias/m
r
$Adam/Hid1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Hid1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/Hid2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_nameAdam/Hid2/kernel/m
{
&Adam/Hid2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Hid2/kernel/m* 
_output_shapes
:
??*
dtype0
y
Adam/Hid2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/Hid2/bias/m
r
$Adam/Hid2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Hid2/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/Hid3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_nameAdam/Hid3/kernel/m
{
&Adam/Hid3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Hid3/kernel/m* 
_output_shapes
:
??*
dtype0
y
Adam/Hid3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/Hid3/bias/m
r
$Adam/Hid3/bias/m/Read/ReadVariableOpReadVariableOpAdam/Hid3/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*%
shared_nameAdam/output/kernel/m
~
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*
_output_shapes
:	?
*
dtype0
|
Adam/output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/output/bias/m
u
&Adam/output/bias/m/Read/ReadVariableOpReadVariableOpAdam/output/bias/m*
_output_shapes
:
*
dtype0
?
Adam/ConA/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/ConA/kernel/v
?
&Adam/ConA/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConA/kernel/v*&
_output_shapes
:*
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
?
Adam/ConB/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/ConB/kernel/v
?
&Adam/ConB/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConB/kernel/v*&
_output_shapes
:*
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
?
Adam/ConA2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConA2/kernel/v
?
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
?
Adam/ConB2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/ConB2/kernel/v
?
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
Adam/ConC/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/ConC/kernel/v
?
&Adam/ConC/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConC/kernel/v*&
_output_shapes
:*
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
?
Adam/ConD/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*#
shared_nameAdam/ConD/kernel/v
?
&Adam/ConD/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConD/kernel/v*&
_output_shapes
:		*
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
?
Adam/ConE/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/ConE/kernel/v
?
&Adam/ConE/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ConE/kernel/v*&
_output_shapes
:*
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
?
Adam/HidA/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_nameAdam/HidA/kernel/v
{
&Adam/HidA/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidA/kernel/v* 
_output_shapes
:
??*
dtype0
y
Adam/HidA/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/HidA/bias/v
r
$Adam/HidA/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidA/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/HidA2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/HidA2/kernel/v
}
'Adam/HidA2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidA2/kernel/v* 
_output_shapes
:
??*
dtype0
{
Adam/HidA2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidA2/bias/v
t
%Adam/HidA2/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidA2/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/HidA3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(?*$
shared_nameAdam/HidA3/kernel/v
|
'Adam/HidA3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidA3/kernel/v*
_output_shapes
:	(?*
dtype0
{
Adam/HidA3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidA3/bias/v
t
%Adam/HidA3/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidA3/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/HidB/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_nameAdam/HidB/kernel/v
{
&Adam/HidB/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidB/kernel/v* 
_output_shapes
:
??*
dtype0
y
Adam/HidB/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/HidB/bias/v
r
$Adam/HidB/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidB/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/HidB2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/HidB2/kernel/v
}
'Adam/HidB2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidB2/kernel/v* 
_output_shapes
:
??*
dtype0
{
Adam/HidB2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidB2/bias/v
t
%Adam/HidB2/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidB2/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/HidB3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(?*$
shared_nameAdam/HidB3/kernel/v
|
'Adam/HidB3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidB3/kernel/v*
_output_shapes
:	(?*
dtype0
{
Adam/HidB3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidB3/bias/v
t
%Adam/HidB3/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidB3/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/HidC/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_nameAdam/HidC/kernel/v
{
&Adam/HidC/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidC/kernel/v* 
_output_shapes
:
??*
dtype0
y
Adam/HidC/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/HidC/bias/v
r
$Adam/HidC/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidC/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/HidD1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/HidD1/kernel/v
}
'Adam/HidD1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidD1/kernel/v* 
_output_shapes
:
??*
dtype0
{
Adam/HidD1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidD1/bias/v
t
%Adam/HidD1/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidD1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/HidE1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/HidE1/kernel/v
}
'Adam/HidE1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HidE1/kernel/v* 
_output_shapes
:
??*
dtype0
{
Adam/HidE1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/HidE1/bias/v
t
%Adam/HidE1/bias/v/Read/ReadVariableOpReadVariableOpAdam/HidE1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/Hid1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?$?*#
shared_nameAdam/Hid1/kernel/v
{
&Adam/Hid1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Hid1/kernel/v* 
_output_shapes
:
?$?*
dtype0
y
Adam/Hid1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/Hid1/bias/v
r
$Adam/Hid1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Hid1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/Hid2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_nameAdam/Hid2/kernel/v
{
&Adam/Hid2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Hid2/kernel/v* 
_output_shapes
:
??*
dtype0
y
Adam/Hid2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/Hid2/bias/v
r
$Adam/Hid2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Hid2/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/Hid3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_nameAdam/Hid3/kernel/v
{
&Adam/Hid3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Hid3/kernel/v* 
_output_shapes
:
??*
dtype0
y
Adam/Hid3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/Hid3/bias/v
r
$Adam/Hid3/bias/v/Read/ReadVariableOpReadVariableOpAdam/Hid3/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*%
shared_nameAdam/output/kernel/v
~
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*
_output_shapes
:	?
*
dtype0
|
Adam/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/output/bias/v
u
&Adam/output/bias/v/Read/ReadVariableOpReadVariableOpAdam/output/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??Bܐ BԐ
?

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

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer-16
layer-17
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
&layer_with_weights-9
&layer-37
'layer_with_weights-10
'layer-38
(layer_with_weights-11
(layer-39
)layer_with_weights-12
)layer-40
*layer_with_weights-13
*layer-41
+layer_with_weights-14
+layer-42
,layer_with_weights-15
,layer-43
-layer_with_weights-16
-layer-44
.layer_with_weights-17
.layer-45
/layer-46
0layer_with_weights-18
0layer-47
1layer_with_weights-19
1layer-48
2layer_with_weights-20
2layer-49
3layer_with_weights-21
3layer-50
4	optimizer
5	variables
6regularization_losses
7trainable_variables
8	keras_api
9
signatures
 
h

:kernel
;bias
<	variables
=regularization_losses
>trainable_variables
?	keras_api
h

@kernel
Abias
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
R
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
R
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
R
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
R
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
h

Vkernel
Wbias
X	variables
Yregularization_losses
Ztrainable_variables
[	keras_api
h

\kernel
]bias
^	variables
_regularization_losses
`trainable_variables
a	keras_api
R
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
R
f	variables
gregularization_losses
htrainable_variables
i	keras_api
h

jkernel
kbias
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
h

pkernel
qbias
r	variables
sregularization_losses
ttrainable_variables
u	keras_api
h

vkernel
wbias
x	variables
yregularization_losses
ztrainable_variables
{	keras_api
j

|kernel
}bias
~	variables
regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate:m?;m?@m?Am?Vm?Wm?\m?]m?jm?km?pm?qm?vm?wm?|m?}m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?:v?;v?@v?Av?Vv?Wv?\v?]v?jv?kv?pv?qv?vv?wv?|v?}v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?
?
:0
;1
@2
A3
V4
W5
\6
]7
j8
k9
p10
q11
v12
w13
|14
}15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
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
 
?
:0
;1
@2
A3
V4
W5
\6
]7
j8
k9
p10
q11
v12
w13
|14
}15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
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
?
5	variables
 ?layer_regularization_losses
?layer_metrics
6regularization_losses
?non_trainable_variables
?layers
?metrics
7trainable_variables
 
WU
VARIABLE_VALUEConA/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	ConA/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

:0
;1
 

:0
;1
?
<	variables
 ?layer_regularization_losses
?layer_metrics
=regularization_losses
?non_trainable_variables
?layers
?metrics
>trainable_variables
WU
VARIABLE_VALUEConB/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	ConB/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

@0
A1
 

@0
A1
?
B	variables
 ?layer_regularization_losses
?layer_metrics
Cregularization_losses
?non_trainable_variables
?layers
?metrics
Dtrainable_variables
 
 
 
?
F	variables
 ?layer_regularization_losses
?layer_metrics
Gregularization_losses
?non_trainable_variables
?layers
?metrics
Htrainable_variables
 
 
 
?
J	variables
 ?layer_regularization_losses
?layer_metrics
Kregularization_losses
?non_trainable_variables
?layers
?metrics
Ltrainable_variables
 
 
 
?
N	variables
 ?layer_regularization_losses
?layer_metrics
Oregularization_losses
?non_trainable_variables
?layers
?metrics
Ptrainable_variables
 
 
 
?
R	variables
 ?layer_regularization_losses
?layer_metrics
Sregularization_losses
?non_trainable_variables
?layers
?metrics
Ttrainable_variables
XV
VARIABLE_VALUEConA2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
ConA2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

V0
W1
 

V0
W1
?
X	variables
 ?layer_regularization_losses
?layer_metrics
Yregularization_losses
?non_trainable_variables
?layers
?metrics
Ztrainable_variables
XV
VARIABLE_VALUEConB2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
ConB2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

\0
]1
 

\0
]1
?
^	variables
 ?layer_regularization_losses
?layer_metrics
_regularization_losses
?non_trainable_variables
?layers
?metrics
`trainable_variables
 
 
 
?
b	variables
 ?layer_regularization_losses
?layer_metrics
cregularization_losses
?non_trainable_variables
?layers
?metrics
dtrainable_variables
 
 
 
?
f	variables
 ?layer_regularization_losses
?layer_metrics
gregularization_losses
?non_trainable_variables
?layers
?metrics
htrainable_variables
XV
VARIABLE_VALUEConA3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
ConA3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

j0
k1
 

j0
k1
?
l	variables
 ?layer_regularization_losses
?layer_metrics
mregularization_losses
?non_trainable_variables
?layers
?metrics
ntrainable_variables
XV
VARIABLE_VALUEConB3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
ConB3/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

p0
q1
 

p0
q1
?
r	variables
 ?layer_regularization_losses
?layer_metrics
sregularization_losses
?non_trainable_variables
?layers
?metrics
ttrainable_variables
WU
VARIABLE_VALUEConC/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	ConC/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

v0
w1
 

v0
w1
?
x	variables
 ?layer_regularization_losses
?layer_metrics
yregularization_losses
?non_trainable_variables
?layers
?metrics
ztrainable_variables
WU
VARIABLE_VALUEConD/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	ConD/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

|0
}1
 

|0
}1
?
~	variables
 ?layer_regularization_losses
?layer_metrics
regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
WU
VARIABLE_VALUEConE/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	ConE/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
 
 
 
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
 
 
 
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
 
 
 
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
 
 
 
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
 
 
 
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
 
 
 
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
 
 
 
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
 
 
 
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
 
 
 
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
 
 
 
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
 
 
 
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
 
 
 
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
 
 
 
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
 
 
 
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
 
 
 
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
 
 
 
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
 
 
 
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
 
 
 
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
 
 
 
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
 
 
 
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
 
 
 
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
WU
VARIABLE_VALUEHidA/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	HidA/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
YW
VARIABLE_VALUEHidA2/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidA2/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
YW
VARIABLE_VALUEHidA3/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidA3/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
XV
VARIABLE_VALUEHidB/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	HidB/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
YW
VARIABLE_VALUEHidB2/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidB2/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
YW
VARIABLE_VALUEHidB3/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidB3/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
XV
VARIABLE_VALUEHidC/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	HidC/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
YW
VARIABLE_VALUEHidD1/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidD1/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
YW
VARIABLE_VALUEHidE1/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
HidE1/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
 
 
 
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
XV
VARIABLE_VALUEHid1/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	Hid1/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
XV
VARIABLE_VALUEHid2/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	Hid2/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
XV
VARIABLE_VALUEHid3/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	Hid3/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
ZX
VARIABLE_VALUEoutput/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEoutput/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
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

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
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
{y
VARIABLE_VALUEAdam/ConB2/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConB2/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConA3/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConA3/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConB3/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConB3/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConC/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConC/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConD/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConD/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConE/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConE/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/HidA/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/HidA/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidA2/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidA2/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidA3/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidA3/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/HidB/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/HidB/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidB2/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidB2/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidB3/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidB3/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/HidC/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/HidC/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidD1/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidD1/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidE1/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidE1/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/Hid1/kernel/mSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Hid1/bias/mQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/Hid2/kernel/mSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Hid2/bias/mQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/Hid3/kernel/mSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Hid3/bias/mQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/output/kernel/mSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/output/bias/mQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
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
{y
VARIABLE_VALUEAdam/ConB2/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConB2/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConA3/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConA3/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/ConB3/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/ConB3/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConC/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConC/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConD/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConD/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/ConE/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/ConE/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/HidA/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/HidA/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidA2/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidA2/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidA3/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidA3/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/HidB/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/HidB/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidB2/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidB2/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidB3/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidB3/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/HidC/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/HidC/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidD1/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidD1/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/HidE1/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/HidE1/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/Hid1/kernel/vSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Hid1/bias/vQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/Hid2/kernel/vSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Hid2/bias/vQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/Hid3/kernel/vSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Hid3/bias/vQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/output/kernel/vSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/output/bias/vQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_inputPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputConB/kernel	ConB/biasConA/kernel	ConA/biasConB2/kernel
ConB2/biasConA2/kernel
ConA2/biasConE/kernel	ConE/biasConD/kernel	ConD/biasConC/kernel	ConC/biasConB3/kernel
ConB3/biasConA3/kernel
ConA3/biasHidA/kernel	HidA/biasHidA2/kernel
HidA2/biasHidA3/kernel
HidA3/biasHidB/kernel	HidB/biasHidB2/kernel
HidB2/biasHidB3/kernel
HidB3/biasHidC/kernel	HidC/biasHidD1/kernel
HidD1/biasHidE1/kernel
HidE1/biasHid1/kernel	Hid1/biasHid2/kernel	Hid2/biasHid3/kernel	Hid3/biasoutput/kerneloutput/bias*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_22545
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?+
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameConA/kernel/Read/ReadVariableOpConA/bias/Read/ReadVariableOpConB/kernel/Read/ReadVariableOpConB/bias/Read/ReadVariableOp ConA2/kernel/Read/ReadVariableOpConA2/bias/Read/ReadVariableOp ConB2/kernel/Read/ReadVariableOpConB2/bias/Read/ReadVariableOp ConA3/kernel/Read/ReadVariableOpConA3/bias/Read/ReadVariableOp ConB3/kernel/Read/ReadVariableOpConB3/bias/Read/ReadVariableOpConC/kernel/Read/ReadVariableOpConC/bias/Read/ReadVariableOpConD/kernel/Read/ReadVariableOpConD/bias/Read/ReadVariableOpConE/kernel/Read/ReadVariableOpConE/bias/Read/ReadVariableOpHidA/kernel/Read/ReadVariableOpHidA/bias/Read/ReadVariableOp HidA2/kernel/Read/ReadVariableOpHidA2/bias/Read/ReadVariableOp HidA3/kernel/Read/ReadVariableOpHidA3/bias/Read/ReadVariableOpHidB/kernel/Read/ReadVariableOpHidB/bias/Read/ReadVariableOp HidB2/kernel/Read/ReadVariableOpHidB2/bias/Read/ReadVariableOp HidB3/kernel/Read/ReadVariableOpHidB3/bias/Read/ReadVariableOpHidC/kernel/Read/ReadVariableOpHidC/bias/Read/ReadVariableOp HidD1/kernel/Read/ReadVariableOpHidD1/bias/Read/ReadVariableOp HidE1/kernel/Read/ReadVariableOpHidE1/bias/Read/ReadVariableOpHid1/kernel/Read/ReadVariableOpHid1/bias/Read/ReadVariableOpHid2/kernel/Read/ReadVariableOpHid2/bias/Read/ReadVariableOpHid3/kernel/Read/ReadVariableOpHid3/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp&Adam/ConA/kernel/m/Read/ReadVariableOp$Adam/ConA/bias/m/Read/ReadVariableOp&Adam/ConB/kernel/m/Read/ReadVariableOp$Adam/ConB/bias/m/Read/ReadVariableOp'Adam/ConA2/kernel/m/Read/ReadVariableOp%Adam/ConA2/bias/m/Read/ReadVariableOp'Adam/ConB2/kernel/m/Read/ReadVariableOp%Adam/ConB2/bias/m/Read/ReadVariableOp'Adam/ConA3/kernel/m/Read/ReadVariableOp%Adam/ConA3/bias/m/Read/ReadVariableOp'Adam/ConB3/kernel/m/Read/ReadVariableOp%Adam/ConB3/bias/m/Read/ReadVariableOp&Adam/ConC/kernel/m/Read/ReadVariableOp$Adam/ConC/bias/m/Read/ReadVariableOp&Adam/ConD/kernel/m/Read/ReadVariableOp$Adam/ConD/bias/m/Read/ReadVariableOp&Adam/ConE/kernel/m/Read/ReadVariableOp$Adam/ConE/bias/m/Read/ReadVariableOp&Adam/HidA/kernel/m/Read/ReadVariableOp$Adam/HidA/bias/m/Read/ReadVariableOp'Adam/HidA2/kernel/m/Read/ReadVariableOp%Adam/HidA2/bias/m/Read/ReadVariableOp'Adam/HidA3/kernel/m/Read/ReadVariableOp%Adam/HidA3/bias/m/Read/ReadVariableOp&Adam/HidB/kernel/m/Read/ReadVariableOp$Adam/HidB/bias/m/Read/ReadVariableOp'Adam/HidB2/kernel/m/Read/ReadVariableOp%Adam/HidB2/bias/m/Read/ReadVariableOp'Adam/HidB3/kernel/m/Read/ReadVariableOp%Adam/HidB3/bias/m/Read/ReadVariableOp&Adam/HidC/kernel/m/Read/ReadVariableOp$Adam/HidC/bias/m/Read/ReadVariableOp'Adam/HidD1/kernel/m/Read/ReadVariableOp%Adam/HidD1/bias/m/Read/ReadVariableOp'Adam/HidE1/kernel/m/Read/ReadVariableOp%Adam/HidE1/bias/m/Read/ReadVariableOp&Adam/Hid1/kernel/m/Read/ReadVariableOp$Adam/Hid1/bias/m/Read/ReadVariableOp&Adam/Hid2/kernel/m/Read/ReadVariableOp$Adam/Hid2/bias/m/Read/ReadVariableOp&Adam/Hid3/kernel/m/Read/ReadVariableOp$Adam/Hid3/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp&Adam/ConA/kernel/v/Read/ReadVariableOp$Adam/ConA/bias/v/Read/ReadVariableOp&Adam/ConB/kernel/v/Read/ReadVariableOp$Adam/ConB/bias/v/Read/ReadVariableOp'Adam/ConA2/kernel/v/Read/ReadVariableOp%Adam/ConA2/bias/v/Read/ReadVariableOp'Adam/ConB2/kernel/v/Read/ReadVariableOp%Adam/ConB2/bias/v/Read/ReadVariableOp'Adam/ConA3/kernel/v/Read/ReadVariableOp%Adam/ConA3/bias/v/Read/ReadVariableOp'Adam/ConB3/kernel/v/Read/ReadVariableOp%Adam/ConB3/bias/v/Read/ReadVariableOp&Adam/ConC/kernel/v/Read/ReadVariableOp$Adam/ConC/bias/v/Read/ReadVariableOp&Adam/ConD/kernel/v/Read/ReadVariableOp$Adam/ConD/bias/v/Read/ReadVariableOp&Adam/ConE/kernel/v/Read/ReadVariableOp$Adam/ConE/bias/v/Read/ReadVariableOp&Adam/HidA/kernel/v/Read/ReadVariableOp$Adam/HidA/bias/v/Read/ReadVariableOp'Adam/HidA2/kernel/v/Read/ReadVariableOp%Adam/HidA2/bias/v/Read/ReadVariableOp'Adam/HidA3/kernel/v/Read/ReadVariableOp%Adam/HidA3/bias/v/Read/ReadVariableOp&Adam/HidB/kernel/v/Read/ReadVariableOp$Adam/HidB/bias/v/Read/ReadVariableOp'Adam/HidB2/kernel/v/Read/ReadVariableOp%Adam/HidB2/bias/v/Read/ReadVariableOp'Adam/HidB3/kernel/v/Read/ReadVariableOp%Adam/HidB3/bias/v/Read/ReadVariableOp&Adam/HidC/kernel/v/Read/ReadVariableOp$Adam/HidC/bias/v/Read/ReadVariableOp'Adam/HidD1/kernel/v/Read/ReadVariableOp%Adam/HidD1/bias/v/Read/ReadVariableOp'Adam/HidE1/kernel/v/Read/ReadVariableOp%Adam/HidE1/bias/v/Read/ReadVariableOp&Adam/Hid1/kernel/v/Read/ReadVariableOp$Adam/Hid1/bias/v/Read/ReadVariableOp&Adam/Hid2/kernel/v/Read/ReadVariableOp$Adam/Hid2/bias/v/Read/ReadVariableOp&Adam/Hid3/kernel/v/Read/ReadVariableOp$Adam/Hid3/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*?
Tin?
?2?	*
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
GPU2*0J 8? *'
f"R 
__inference__traced_save_24621
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConA/kernel	ConA/biasConB/kernel	ConB/biasConA2/kernel
ConA2/biasConB2/kernel
ConB2/biasConA3/kernel
ConA3/biasConB3/kernel
ConB3/biasConC/kernel	ConC/biasConD/kernel	ConD/biasConE/kernel	ConE/biasHidA/kernel	HidA/biasHidA2/kernel
HidA2/biasHidA3/kernel
HidA3/biasHidB/kernel	HidB/biasHidB2/kernel
HidB2/biasHidB3/kernel
HidB3/biasHidC/kernel	HidC/biasHidD1/kernel
HidD1/biasHidE1/kernel
HidE1/biasHid1/kernel	Hid1/biasHid2/kernel	Hid2/biasHid3/kernel	Hid3/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/ConA/kernel/mAdam/ConA/bias/mAdam/ConB/kernel/mAdam/ConB/bias/mAdam/ConA2/kernel/mAdam/ConA2/bias/mAdam/ConB2/kernel/mAdam/ConB2/bias/mAdam/ConA3/kernel/mAdam/ConA3/bias/mAdam/ConB3/kernel/mAdam/ConB3/bias/mAdam/ConC/kernel/mAdam/ConC/bias/mAdam/ConD/kernel/mAdam/ConD/bias/mAdam/ConE/kernel/mAdam/ConE/bias/mAdam/HidA/kernel/mAdam/HidA/bias/mAdam/HidA2/kernel/mAdam/HidA2/bias/mAdam/HidA3/kernel/mAdam/HidA3/bias/mAdam/HidB/kernel/mAdam/HidB/bias/mAdam/HidB2/kernel/mAdam/HidB2/bias/mAdam/HidB3/kernel/mAdam/HidB3/bias/mAdam/HidC/kernel/mAdam/HidC/bias/mAdam/HidD1/kernel/mAdam/HidD1/bias/mAdam/HidE1/kernel/mAdam/HidE1/bias/mAdam/Hid1/kernel/mAdam/Hid1/bias/mAdam/Hid2/kernel/mAdam/Hid2/bias/mAdam/Hid3/kernel/mAdam/Hid3/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/ConA/kernel/vAdam/ConA/bias/vAdam/ConB/kernel/vAdam/ConB/bias/vAdam/ConA2/kernel/vAdam/ConA2/bias/vAdam/ConB2/kernel/vAdam/ConB2/bias/vAdam/ConA3/kernel/vAdam/ConA3/bias/vAdam/ConB3/kernel/vAdam/ConB3/bias/vAdam/ConC/kernel/vAdam/ConC/bias/vAdam/ConD/kernel/vAdam/ConD/bias/vAdam/ConE/kernel/vAdam/ConE/bias/vAdam/HidA/kernel/vAdam/HidA/bias/vAdam/HidA2/kernel/vAdam/HidA2/bias/vAdam/HidA3/kernel/vAdam/HidA3/bias/vAdam/HidB/kernel/vAdam/HidB/bias/vAdam/HidB2/kernel/vAdam/HidB2/bias/vAdam/HidB3/kernel/vAdam/HidB3/bias/vAdam/HidC/kernel/vAdam/HidC/bias/vAdam/HidD1/kernel/vAdam/HidD1/bias/vAdam/HidE1/kernel/vAdam/HidE1/bias/vAdam/Hid1/kernel/vAdam/Hid1/bias/vAdam/Hid2/kernel/vAdam/Hid2/bias/vAdam/Hid3/kernel/vAdam/Hid3/bias/vAdam/output/kernel/vAdam/output/bias/v*?
Tin?
?2?*
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
GPU2*0J 8? **
f%R#
!__inference__traced_restore_25054?? 
?
]
A__inference_PoolB2_layer_call_and_return_conditional_losses_20626

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????		:W S
/
_output_shapes
:?????????		
 
_user_specified_nameinputs
?
?
@__inference_HidB2_layer_call_and_return_conditional_losses_20949

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_concatenate_layer_call_and_return_conditional_losses_24095
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8concat/axis:output:0*
N	*
T0*(
_output_shapes
:??????????$2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/2:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/3:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/4:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/5:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/6:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/7:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/8
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_23723

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????

2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????

2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_22160	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:		

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:(

unknown_14:($

unknown_15:(

unknown_16:(

unknown_17:
??

unknown_18:	?

unknown_19:
??

unknown_20:	?

unknown_21:	(?

unknown_22:	?

unknown_23:
??

unknown_24:	?

unknown_25:
??

unknown_26:	?

unknown_27:	(?

unknown_28:	?

unknown_29:
??

unknown_30:	?

unknown_31:
??

unknown_32:	?

unknown_33:
??

unknown_34:	?

unknown_35:
?$?

unknown_36:	?

unknown_37:
??

unknown_38:	?

unknown_39:
??

unknown_40:	?

unknown_41:	?


unknown_42:

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
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_219762
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
??
?
@__inference_model_layer_call_and_return_conditional_losses_22444	
input$

conb_22305:

conb_22307:$

cona_22310:

cona_22312:%
conb2_22319:
conb2_22321:%
cona2_22324:
cona2_22326:$

cone_22331:

cone_22333:$

cond_22336:		

cond_22338:$

conc_22341:

conc_22343:%
conb3_22346:(
conb3_22348:(%
cona3_22351:(
cona3_22353:(

hida_22377:
??

hida_22379:	?
hida2_22382:
??
hida2_22384:	?
hida3_22387:	(?
hida3_22389:	?

hidb_22392:
??

hidb_22394:	?
hidb2_22397:
??
hidb2_22399:	?
hidb3_22402:	(?
hidb3_22404:	?

hidc_22407:
??

hidc_22409:	?
hidd1_22412:
??
hidd1_22414:	?
hide1_22417:
??
hide1_22419:	?

hid1_22423:
?$?

hid1_22425:	?

hid2_22428:
??

hid2_22430:	?

hid3_22433:
??

hid3_22435:	?
output_22438:	?

output_22440:

identity??ConA/StatefulPartitionedCall?ConA2/StatefulPartitionedCall?ConA3/StatefulPartitionedCall?ConB/StatefulPartitionedCall?ConB2/StatefulPartitionedCall?ConB3/StatefulPartitionedCall?ConC/StatefulPartitionedCall?ConD/StatefulPartitionedCall?ConE/StatefulPartitionedCall?Hid1/StatefulPartitionedCall?Hid2/StatefulPartitionedCall?Hid3/StatefulPartitionedCall?HidA/StatefulPartitionedCall?HidA2/StatefulPartitionedCall?HidA3/StatefulPartitionedCall?HidB/StatefulPartitionedCall?HidB2/StatefulPartitionedCall?HidB3/StatefulPartitionedCall?HidC/StatefulPartitionedCall?HidD1/StatefulPartitionedCall?HidE1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?!dropout_5/StatefulPartitionedCall?!dropout_6/StatefulPartitionedCall?!dropout_7/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?output/StatefulPartitionedCall?
ConB/StatefulPartitionedCallStatefulPartitionedCallinput
conb_22305
conb_22307*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_ConB_layer_call_and_return_conditional_losses_205392
ConB/StatefulPartitionedCall?
ConA/StatefulPartitionedCallStatefulPartitionedCallinput
cona_22310
cona_22312*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_ConA_layer_call_and_return_conditional_losses_205562
ConA/StatefulPartitionedCall?
PoolB/PartitionedCallPartitionedCall%ConB/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_PoolB_layer_call_and_return_conditional_losses_205662
PoolB/PartitionedCall?
PoolA/PartitionedCallPartitionedCall%ConA/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_PoolA_layer_call_and_return_conditional_losses_205722
PoolA/PartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCallPoolB/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_217062#
!dropout_1/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCallPoolA/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_216832!
dropout/StatefulPartitionedCall?
ConB2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conb2_22319conb2_22321*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConB2_layer_call_and_return_conditional_losses_205992
ConB2/StatefulPartitionedCall?
ConA2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0cona2_22324cona2_22326*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConA2_layer_call_and_return_conditional_losses_206162
ConA2/StatefulPartitionedCall?
PoolB2/PartitionedCallPartitionedCall&ConB2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolB2_layer_call_and_return_conditional_losses_206262
PoolB2/PartitionedCall?
PoolA2/PartitionedCallPartitionedCall&ConA2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolA2_layer_call_and_return_conditional_losses_206322
PoolA2/PartitionedCall?
ConE/StatefulPartitionedCallStatefulPartitionedCallinput
cone_22331
cone_22333*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_ConE_layer_call_and_return_conditional_losses_206452
ConE/StatefulPartitionedCall?
ConD/StatefulPartitionedCallStatefulPartitionedCallinput
cond_22336
cond_22338*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_ConD_layer_call_and_return_conditional_losses_206622
ConD/StatefulPartitionedCall?
ConC/StatefulPartitionedCallStatefulPartitionedCallinput
conc_22341
conc_22343*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_ConC_layer_call_and_return_conditional_losses_206792
ConC/StatefulPartitionedCall?
ConB3/StatefulPartitionedCallStatefulPartitionedCallPoolB2/PartitionedCall:output:0conb3_22346conb3_22348*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConB3_layer_call_and_return_conditional_losses_206962
ConB3/StatefulPartitionedCall?
ConA3/StatefulPartitionedCallStatefulPartitionedCallPoolA2/PartitionedCall:output:0cona3_22351cona3_22353*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConA3_layer_call_and_return_conditional_losses_207132
ConA3/StatefulPartitionedCall?
PoolE/PartitionedCallPartitionedCall%ConE/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_PoolE_layer_call_and_return_conditional_losses_207232
PoolE/PartitionedCall?
PoolD/PartitionedCallPartitionedCall%ConD/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_PoolD_layer_call_and_return_conditional_losses_207292
PoolD/PartitionedCall?
PoolC/PartitionedCallPartitionedCall%ConC/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_PoolC_layer_call_and_return_conditional_losses_207352
PoolC/PartitionedCall?
PoolB3/PartitionedCallPartitionedCall&ConB3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolB3_layer_call_and_return_conditional_losses_207412
PoolB3/PartitionedCall?
PoolA3/PartitionedCallPartitionedCall&ConA3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolA3_layer_call_and_return_conditional_losses_207472
PoolA3/PartitionedCall?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCallPoolE/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_215552#
!dropout_4/StatefulPartitionedCall?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCallPoolD/PartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_215322#
!dropout_3/StatefulPartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCallPoolC/PartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_215092#
!dropout_2/StatefulPartitionedCall?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCallPoolB3/PartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_214862#
!dropout_8/StatefulPartitionedCall?
!dropout_7/StatefulPartitionedCallStatefulPartitionedCallPoolB2/PartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_214632#
!dropout_7/StatefulPartitionedCall?
!dropout_6/StatefulPartitionedCallStatefulPartitionedCallPoolA3/PartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_214402#
!dropout_6/StatefulPartitionedCall?
!dropout_5/StatefulPartitionedCallStatefulPartitionedCallPoolA2/PartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_214172#
!dropout_5/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCall*dropout_4/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *M
fHRF
D__inference_flatten_8_layer_call_and_return_conditional_losses_208042
flatten_8/PartitionedCall?
flatten_7/PartitionedCallPartitionedCall*dropout_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_7_layer_call_and_return_conditional_losses_208122
flatten_7/PartitionedCall?
flatten_6/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_208202
flatten_6/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall*dropout_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_208282
flatten_5/PartitionedCall?
flatten_4/PartitionedCallPartitionedCall*dropout_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_208362
flatten_4/PartitionedCall?
flatten_3/PartitionedCallPartitionedCallPoolB/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_208442
flatten_3/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall*dropout_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_208522
flatten_2/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall*dropout_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_208602
flatten_1/PartitionedCall?
flatten/PartitionedCallPartitionedCallPoolA/PartitionedCall:output:0*
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
GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_208682
flatten/PartitionedCall?
HidA/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
hida_22377
hida_22379*
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
GPU2*0J 8? *H
fCRA
?__inference_HidA_layer_call_and_return_conditional_losses_208812
HidA/StatefulPartitionedCall?
HidA2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0hida2_22382hida2_22384*
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
@__inference_HidA2_layer_call_and_return_conditional_losses_208982
HidA2/StatefulPartitionedCall?
HidA3/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0hida3_22387hida3_22389*
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
@__inference_HidA3_layer_call_and_return_conditional_losses_209152
HidA3/StatefulPartitionedCall?
HidB/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0
hidb_22392
hidb_22394*
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
GPU2*0J 8? *H
fCRA
?__inference_HidB_layer_call_and_return_conditional_losses_209322
HidB/StatefulPartitionedCall?
HidB2/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0hidb2_22397hidb2_22399*
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
@__inference_HidB2_layer_call_and_return_conditional_losses_209492
HidB2/StatefulPartitionedCall?
HidB3/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0hidb3_22402hidb3_22404*
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
@__inference_HidB3_layer_call_and_return_conditional_losses_209662
HidB3/StatefulPartitionedCall?
HidC/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0
hidc_22407
hidc_22409*
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
GPU2*0J 8? *H
fCRA
?__inference_HidC_layer_call_and_return_conditional_losses_209832
HidC/StatefulPartitionedCall?
HidD1/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0hidd1_22412hidd1_22414*
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
@__inference_HidD1_layer_call_and_return_conditional_losses_210002
HidD1/StatefulPartitionedCall?
HidE1/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0hide1_22417hide1_22419*
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
@__inference_HidE1_layer_call_and_return_conditional_losses_210172
HidE1/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall%HidA/StatefulPartitionedCall:output:0&HidA2/StatefulPartitionedCall:output:0&HidA3/StatefulPartitionedCall:output:0%HidB/StatefulPartitionedCall:output:0&HidB2/StatefulPartitionedCall:output:0&HidB3/StatefulPartitionedCall:output:0%HidC/StatefulPartitionedCall:output:0&HidD1/StatefulPartitionedCall:output:0&HidE1/StatefulPartitionedCall:output:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_210372
concatenate/PartitionedCall?
Hid1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0
hid1_22423
hid1_22425*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_Hid1_layer_call_and_return_conditional_losses_210502
Hid1/StatefulPartitionedCall?
Hid2/StatefulPartitionedCallStatefulPartitionedCall%Hid1/StatefulPartitionedCall:output:0
hid2_22428
hid2_22430*
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
GPU2*0J 8? *H
fCRA
?__inference_Hid2_layer_call_and_return_conditional_losses_210672
Hid2/StatefulPartitionedCall?
Hid3/StatefulPartitionedCallStatefulPartitionedCall%Hid2/StatefulPartitionedCall:output:0
hid3_22433
hid3_22435*
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
GPU2*0J 8? *H
fCRA
?__inference_Hid3_layer_call_and_return_conditional_losses_210842
Hid3/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall%Hid3/StatefulPartitionedCall:output:0output_22438output_22440*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_211012 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp^ConA/StatefulPartitionedCall^ConA2/StatefulPartitionedCall^ConA3/StatefulPartitionedCall^ConB/StatefulPartitionedCall^ConB2/StatefulPartitionedCall^ConB3/StatefulPartitionedCall^ConC/StatefulPartitionedCall^ConD/StatefulPartitionedCall^ConE/StatefulPartitionedCall^Hid1/StatefulPartitionedCall^Hid2/StatefulPartitionedCall^Hid3/StatefulPartitionedCall^HidA/StatefulPartitionedCall^HidA2/StatefulPartitionedCall^HidA3/StatefulPartitionedCall^HidB/StatefulPartitionedCall^HidB2/StatefulPartitionedCall^HidB3/StatefulPartitionedCall^HidC/StatefulPartitionedCall^HidD1/StatefulPartitionedCall^HidE1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
ConA/StatefulPartitionedCallConA/StatefulPartitionedCall2>
ConA2/StatefulPartitionedCallConA2/StatefulPartitionedCall2>
ConA3/StatefulPartitionedCallConA3/StatefulPartitionedCall2<
ConB/StatefulPartitionedCallConB/StatefulPartitionedCall2>
ConB2/StatefulPartitionedCallConB2/StatefulPartitionedCall2>
ConB3/StatefulPartitionedCallConB3/StatefulPartitionedCall2<
ConC/StatefulPartitionedCallConC/StatefulPartitionedCall2<
ConD/StatefulPartitionedCallConD/StatefulPartitionedCall2<
ConE/StatefulPartitionedCallConE/StatefulPartitionedCall2<
Hid1/StatefulPartitionedCallHid1/StatefulPartitionedCall2<
Hid2/StatefulPartitionedCallHid2/StatefulPartitionedCall2<
Hid3/StatefulPartitionedCallHid3/StatefulPartitionedCall2<
HidA/StatefulPartitionedCallHidA/StatefulPartitionedCall2>
HidA2/StatefulPartitionedCallHidA2/StatefulPartitionedCall2>
HidA3/StatefulPartitionedCallHidA3/StatefulPartitionedCall2<
HidB/StatefulPartitionedCallHidB/StatefulPartitionedCall2>
HidB2/StatefulPartitionedCallHidB2/StatefulPartitionedCall2>
HidB3/StatefulPartitionedCallHidB3/StatefulPartitionedCall2<
HidC/StatefulPartitionedCallHidC/StatefulPartitionedCall2>
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
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
??
?
@__inference_model_layer_call_and_return_conditional_losses_21108

inputs$

conb_20540:

conb_20542:$

cona_20557:

cona_20559:%
conb2_20600:
conb2_20602:%
cona2_20617:
cona2_20619:$

cone_20646:

cone_20648:$

cond_20663:		

cond_20665:$

conc_20680:

conc_20682:%
conb3_20697:(
conb3_20699:(%
cona3_20714:(
cona3_20716:(

hida_20882:
??

hida_20884:	?
hida2_20899:
??
hida2_20901:	?
hida3_20916:	(?
hida3_20918:	?

hidb_20933:
??

hidb_20935:	?
hidb2_20950:
??
hidb2_20952:	?
hidb3_20967:	(?
hidb3_20969:	?

hidc_20984:
??

hidc_20986:	?
hidd1_21001:
??
hidd1_21003:	?
hide1_21018:
??
hide1_21020:	?

hid1_21051:
?$?

hid1_21053:	?

hid2_21068:
??

hid2_21070:	?

hid3_21085:
??

hid3_21087:	?
output_21102:	?

output_21104:

identity??ConA/StatefulPartitionedCall?ConA2/StatefulPartitionedCall?ConA3/StatefulPartitionedCall?ConB/StatefulPartitionedCall?ConB2/StatefulPartitionedCall?ConB3/StatefulPartitionedCall?ConC/StatefulPartitionedCall?ConD/StatefulPartitionedCall?ConE/StatefulPartitionedCall?Hid1/StatefulPartitionedCall?Hid2/StatefulPartitionedCall?Hid3/StatefulPartitionedCall?HidA/StatefulPartitionedCall?HidA2/StatefulPartitionedCall?HidA3/StatefulPartitionedCall?HidB/StatefulPartitionedCall?HidB2/StatefulPartitionedCall?HidB3/StatefulPartitionedCall?HidC/StatefulPartitionedCall?HidD1/StatefulPartitionedCall?HidE1/StatefulPartitionedCall?output/StatefulPartitionedCall?
ConB/StatefulPartitionedCallStatefulPartitionedCallinputs
conb_20540
conb_20542*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_ConB_layer_call_and_return_conditional_losses_205392
ConB/StatefulPartitionedCall?
ConA/StatefulPartitionedCallStatefulPartitionedCallinputs
cona_20557
cona_20559*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_ConA_layer_call_and_return_conditional_losses_205562
ConA/StatefulPartitionedCall?
PoolB/PartitionedCallPartitionedCall%ConB/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_PoolB_layer_call_and_return_conditional_losses_205662
PoolB/PartitionedCall?
PoolA/PartitionedCallPartitionedCall%ConA/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_PoolA_layer_call_and_return_conditional_losses_205722
PoolA/PartitionedCall?
dropout_1/PartitionedCallPartitionedCallPoolB/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_205792
dropout_1/PartitionedCall?
dropout/PartitionedCallPartitionedCallPoolA/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_205862
dropout/PartitionedCall?
ConB2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conb2_20600conb2_20602*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConB2_layer_call_and_return_conditional_losses_205992
ConB2/StatefulPartitionedCall?
ConA2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0cona2_20617cona2_20619*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConA2_layer_call_and_return_conditional_losses_206162
ConA2/StatefulPartitionedCall?
PoolB2/PartitionedCallPartitionedCall&ConB2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolB2_layer_call_and_return_conditional_losses_206262
PoolB2/PartitionedCall?
PoolA2/PartitionedCallPartitionedCall&ConA2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolA2_layer_call_and_return_conditional_losses_206322
PoolA2/PartitionedCall?
ConE/StatefulPartitionedCallStatefulPartitionedCallinputs
cone_20646
cone_20648*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_ConE_layer_call_and_return_conditional_losses_206452
ConE/StatefulPartitionedCall?
ConD/StatefulPartitionedCallStatefulPartitionedCallinputs
cond_20663
cond_20665*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_ConD_layer_call_and_return_conditional_losses_206622
ConD/StatefulPartitionedCall?
ConC/StatefulPartitionedCallStatefulPartitionedCallinputs
conc_20680
conc_20682*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_ConC_layer_call_and_return_conditional_losses_206792
ConC/StatefulPartitionedCall?
ConB3/StatefulPartitionedCallStatefulPartitionedCallPoolB2/PartitionedCall:output:0conb3_20697conb3_20699*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConB3_layer_call_and_return_conditional_losses_206962
ConB3/StatefulPartitionedCall?
ConA3/StatefulPartitionedCallStatefulPartitionedCallPoolA2/PartitionedCall:output:0cona3_20714cona3_20716*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConA3_layer_call_and_return_conditional_losses_207132
ConA3/StatefulPartitionedCall?
PoolE/PartitionedCallPartitionedCall%ConE/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_PoolE_layer_call_and_return_conditional_losses_207232
PoolE/PartitionedCall?
PoolD/PartitionedCallPartitionedCall%ConD/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_PoolD_layer_call_and_return_conditional_losses_207292
PoolD/PartitionedCall?
PoolC/PartitionedCallPartitionedCall%ConC/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_PoolC_layer_call_and_return_conditional_losses_207352
PoolC/PartitionedCall?
PoolB3/PartitionedCallPartitionedCall&ConB3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolB3_layer_call_and_return_conditional_losses_207412
PoolB3/PartitionedCall?
PoolA3/PartitionedCallPartitionedCall&ConA3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolA3_layer_call_and_return_conditional_losses_207472
PoolA3/PartitionedCall?
dropout_4/PartitionedCallPartitionedCallPoolE/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_207542
dropout_4/PartitionedCall?
dropout_3/PartitionedCallPartitionedCallPoolD/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_207612
dropout_3/PartitionedCall?
dropout_2/PartitionedCallPartitionedCallPoolC/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_207682
dropout_2/PartitionedCall?
dropout_8/PartitionedCallPartitionedCallPoolB3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_207752
dropout_8/PartitionedCall?
dropout_7/PartitionedCallPartitionedCallPoolB2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_207822
dropout_7/PartitionedCall?
dropout_6/PartitionedCallPartitionedCallPoolA3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_207892
dropout_6/PartitionedCall?
dropout_5/PartitionedCallPartitionedCallPoolA2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_207962
dropout_5/PartitionedCall?
flatten_8/PartitionedCallPartitionedCall"dropout_4/PartitionedCall:output:0*
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
GPU2*0J 8? *M
fHRF
D__inference_flatten_8_layer_call_and_return_conditional_losses_208042
flatten_8/PartitionedCall?
flatten_7/PartitionedCallPartitionedCall"dropout_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_7_layer_call_and_return_conditional_losses_208122
flatten_7/PartitionedCall?
flatten_6/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_208202
flatten_6/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall"dropout_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_208282
flatten_5/PartitionedCall?
flatten_4/PartitionedCallPartitionedCall"dropout_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_208362
flatten_4/PartitionedCall?
flatten_3/PartitionedCallPartitionedCallPoolB/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_208442
flatten_3/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall"dropout_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_208522
flatten_2/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall"dropout_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_208602
flatten_1/PartitionedCall?
flatten/PartitionedCallPartitionedCallPoolA/PartitionedCall:output:0*
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
GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_208682
flatten/PartitionedCall?
HidA/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
hida_20882
hida_20884*
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
GPU2*0J 8? *H
fCRA
?__inference_HidA_layer_call_and_return_conditional_losses_208812
HidA/StatefulPartitionedCall?
HidA2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0hida2_20899hida2_20901*
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
@__inference_HidA2_layer_call_and_return_conditional_losses_208982
HidA2/StatefulPartitionedCall?
HidA3/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0hida3_20916hida3_20918*
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
@__inference_HidA3_layer_call_and_return_conditional_losses_209152
HidA3/StatefulPartitionedCall?
HidB/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0
hidb_20933
hidb_20935*
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
GPU2*0J 8? *H
fCRA
?__inference_HidB_layer_call_and_return_conditional_losses_209322
HidB/StatefulPartitionedCall?
HidB2/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0hidb2_20950hidb2_20952*
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
@__inference_HidB2_layer_call_and_return_conditional_losses_209492
HidB2/StatefulPartitionedCall?
HidB3/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0hidb3_20967hidb3_20969*
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
@__inference_HidB3_layer_call_and_return_conditional_losses_209662
HidB3/StatefulPartitionedCall?
HidC/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0
hidc_20984
hidc_20986*
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
GPU2*0J 8? *H
fCRA
?__inference_HidC_layer_call_and_return_conditional_losses_209832
HidC/StatefulPartitionedCall?
HidD1/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0hidd1_21001hidd1_21003*
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
@__inference_HidD1_layer_call_and_return_conditional_losses_210002
HidD1/StatefulPartitionedCall?
HidE1/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0hide1_21018hide1_21020*
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
@__inference_HidE1_layer_call_and_return_conditional_losses_210172
HidE1/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall%HidA/StatefulPartitionedCall:output:0&HidA2/StatefulPartitionedCall:output:0&HidA3/StatefulPartitionedCall:output:0%HidB/StatefulPartitionedCall:output:0&HidB2/StatefulPartitionedCall:output:0&HidB3/StatefulPartitionedCall:output:0%HidC/StatefulPartitionedCall:output:0&HidD1/StatefulPartitionedCall:output:0&HidE1/StatefulPartitionedCall:output:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_210372
concatenate/PartitionedCall?
Hid1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0
hid1_21051
hid1_21053*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_Hid1_layer_call_and_return_conditional_losses_210502
Hid1/StatefulPartitionedCall?
Hid2/StatefulPartitionedCallStatefulPartitionedCall%Hid1/StatefulPartitionedCall:output:0
hid2_21068
hid2_21070*
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
GPU2*0J 8? *H
fCRA
?__inference_Hid2_layer_call_and_return_conditional_losses_210672
Hid2/StatefulPartitionedCall?
Hid3/StatefulPartitionedCallStatefulPartitionedCall%Hid2/StatefulPartitionedCall:output:0
hid3_21085
hid3_21087*
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
GPU2*0J 8? *H
fCRA
?__inference_Hid3_layer_call_and_return_conditional_losses_210842
Hid3/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall%Hid3/StatefulPartitionedCall:output:0output_21102output_21104*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_211012 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp^ConA/StatefulPartitionedCall^ConA2/StatefulPartitionedCall^ConA3/StatefulPartitionedCall^ConB/StatefulPartitionedCall^ConB2/StatefulPartitionedCall^ConB3/StatefulPartitionedCall^ConC/StatefulPartitionedCall^ConD/StatefulPartitionedCall^ConE/StatefulPartitionedCall^Hid1/StatefulPartitionedCall^Hid2/StatefulPartitionedCall^Hid3/StatefulPartitionedCall^HidA/StatefulPartitionedCall^HidA2/StatefulPartitionedCall^HidA3/StatefulPartitionedCall^HidB/StatefulPartitionedCall^HidB2/StatefulPartitionedCall^HidB3/StatefulPartitionedCall^HidC/StatefulPartitionedCall^HidD1/StatefulPartitionedCall^HidE1/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
ConA/StatefulPartitionedCallConA/StatefulPartitionedCall2>
ConA2/StatefulPartitionedCallConA2/StatefulPartitionedCall2>
ConA3/StatefulPartitionedCallConA3/StatefulPartitionedCall2<
ConB/StatefulPartitionedCallConB/StatefulPartitionedCall2>
ConB2/StatefulPartitionedCallConB2/StatefulPartitionedCall2>
ConB3/StatefulPartitionedCallConB3/StatefulPartitionedCall2<
ConC/StatefulPartitionedCallConC/StatefulPartitionedCall2<
ConD/StatefulPartitionedCallConD/StatefulPartitionedCall2<
ConE/StatefulPartitionedCallConE/StatefulPartitionedCall2<
Hid1/StatefulPartitionedCallHid1/StatefulPartitionedCall2<
Hid2/StatefulPartitionedCallHid2/StatefulPartitionedCall2<
Hid3/StatefulPartitionedCallHid3/StatefulPartitionedCall2<
HidA/StatefulPartitionedCallHidA/StatefulPartitionedCall2>
HidA2/StatefulPartitionedCallHidA2/StatefulPartitionedCall2>
HidA3/StatefulPartitionedCallHidA3/StatefulPartitionedCall2<
HidB/StatefulPartitionedCallHidB/StatefulPartitionedCall2>
HidB2/StatefulPartitionedCallHidB2/StatefulPartitionedCall2>
HidB3/StatefulPartitionedCallHidB3/StatefulPartitionedCall2<
HidC/StatefulPartitionedCallHidC/StatefulPartitionedCall2>
HidD1/StatefulPartitionedCallHidD1/StatefulPartitionedCall2>
HidE1/StatefulPartitionedCallHidE1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_23320

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
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
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
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
]
A__inference_PoolA2_layer_call_and_return_conditional_losses_23375

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
E
)__inference_dropout_3_layer_call_fn_23740

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
:?????????		* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_207612
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????		2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????		:W S
/
_output_shapes
:?????????		
 
_user_specified_nameinputs
?
A
%__inference_PoolB_layer_call_fn_23256

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_PoolB_layer_call_and_return_conditional_losses_205662
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_7_layer_call_and_return_conditional_losses_23681

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
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
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
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
?__inference_ConA_layer_call_and_return_conditional_losses_23206

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
@__inference_model_layer_call_and_return_conditional_losses_23186

inputs=
#conb_conv2d_readvariableop_resource:2
$conb_biasadd_readvariableop_resource:=
#cona_conv2d_readvariableop_resource:2
$cona_biasadd_readvariableop_resource:>
$conb2_conv2d_readvariableop_resource:3
%conb2_biasadd_readvariableop_resource:>
$cona2_conv2d_readvariableop_resource:3
%cona2_biasadd_readvariableop_resource:=
#cone_conv2d_readvariableop_resource:2
$cone_biasadd_readvariableop_resource:=
#cond_conv2d_readvariableop_resource:		2
$cond_biasadd_readvariableop_resource:=
#conc_conv2d_readvariableop_resource:2
$conc_biasadd_readvariableop_resource:>
$conb3_conv2d_readvariableop_resource:(3
%conb3_biasadd_readvariableop_resource:(>
$cona3_conv2d_readvariableop_resource:(3
%cona3_biasadd_readvariableop_resource:(7
#hida_matmul_readvariableop_resource:
??3
$hida_biasadd_readvariableop_resource:	?8
$hida2_matmul_readvariableop_resource:
??4
%hida2_biasadd_readvariableop_resource:	?7
$hida3_matmul_readvariableop_resource:	(?4
%hida3_biasadd_readvariableop_resource:	?7
#hidb_matmul_readvariableop_resource:
??3
$hidb_biasadd_readvariableop_resource:	?8
$hidb2_matmul_readvariableop_resource:
??4
%hidb2_biasadd_readvariableop_resource:	?7
$hidb3_matmul_readvariableop_resource:	(?4
%hidb3_biasadd_readvariableop_resource:	?7
#hidc_matmul_readvariableop_resource:
??3
$hidc_biasadd_readvariableop_resource:	?8
$hidd1_matmul_readvariableop_resource:
??4
%hidd1_biasadd_readvariableop_resource:	?8
$hide1_matmul_readvariableop_resource:
??4
%hide1_biasadd_readvariableop_resource:	?7
#hid1_matmul_readvariableop_resource:
?$?3
$hid1_biasadd_readvariableop_resource:	?7
#hid2_matmul_readvariableop_resource:
??3
$hid2_biasadd_readvariableop_resource:	?7
#hid3_matmul_readvariableop_resource:
??3
$hid3_biasadd_readvariableop_resource:	?8
%output_matmul_readvariableop_resource:	?
4
&output_biasadd_readvariableop_resource:

identity??ConA/BiasAdd/ReadVariableOp?ConA/Conv2D/ReadVariableOp?ConA2/BiasAdd/ReadVariableOp?ConA2/Conv2D/ReadVariableOp?ConA3/BiasAdd/ReadVariableOp?ConA3/Conv2D/ReadVariableOp?ConB/BiasAdd/ReadVariableOp?ConB/Conv2D/ReadVariableOp?ConB2/BiasAdd/ReadVariableOp?ConB2/Conv2D/ReadVariableOp?ConB3/BiasAdd/ReadVariableOp?ConB3/Conv2D/ReadVariableOp?ConC/BiasAdd/ReadVariableOp?ConC/Conv2D/ReadVariableOp?ConD/BiasAdd/ReadVariableOp?ConD/Conv2D/ReadVariableOp?ConE/BiasAdd/ReadVariableOp?ConE/Conv2D/ReadVariableOp?Hid1/BiasAdd/ReadVariableOp?Hid1/MatMul/ReadVariableOp?Hid2/BiasAdd/ReadVariableOp?Hid2/MatMul/ReadVariableOp?Hid3/BiasAdd/ReadVariableOp?Hid3/MatMul/ReadVariableOp?HidA/BiasAdd/ReadVariableOp?HidA/MatMul/ReadVariableOp?HidA2/BiasAdd/ReadVariableOp?HidA2/MatMul/ReadVariableOp?HidA3/BiasAdd/ReadVariableOp?HidA3/MatMul/ReadVariableOp?HidB/BiasAdd/ReadVariableOp?HidB/MatMul/ReadVariableOp?HidB2/BiasAdd/ReadVariableOp?HidB2/MatMul/ReadVariableOp?HidB3/BiasAdd/ReadVariableOp?HidB3/MatMul/ReadVariableOp?HidC/BiasAdd/ReadVariableOp?HidC/MatMul/ReadVariableOp?HidD1/BiasAdd/ReadVariableOp?HidD1/MatMul/ReadVariableOp?HidE1/BiasAdd/ReadVariableOp?HidE1/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
ConB/Conv2D/ReadVariableOpReadVariableOp#conb_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConB/Conv2D/ReadVariableOp?
ConB/Conv2DConv2Dinputs"ConB/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
ConB/Conv2D?
ConB/BiasAdd/ReadVariableOpReadVariableOp$conb_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConB/BiasAdd/ReadVariableOp?
ConB/BiasAddBiasAddConB/Conv2D:output:0#ConB/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
ConB/BiasAddo
	ConB/ReluReluConB/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
	ConB/Relu?
ConA/Conv2D/ReadVariableOpReadVariableOp#cona_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConA/Conv2D/ReadVariableOp?
ConA/Conv2DConv2Dinputs"ConA/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
ConA/Conv2D?
ConA/BiasAdd/ReadVariableOpReadVariableOp$cona_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConA/BiasAdd/ReadVariableOp?
ConA/BiasAddBiasAddConA/Conv2D:output:0#ConA/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
ConA/BiasAddo
	ConA/ReluReluConA/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
	ConA/Relu?
PoolB/MaxPoolMaxPoolConB/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
PoolB/MaxPool?
PoolA/MaxPoolMaxPoolConA/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
PoolA/MaxPoolw
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
:?????????2
dropout_1/dropout/Mulx
dropout_1/dropout/ShapeShapePoolB/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
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
:?????????2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
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
:?????????2
dropout/dropout/Mult
dropout/dropout/ShapeShapePoolA/MaxPool:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
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
:?????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/dropout/Mul_1?
ConB2/Conv2D/ReadVariableOpReadVariableOp$conb2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConB2/Conv2D/ReadVariableOp?
ConB2/Conv2DConv2Ddropout_1/dropout/Mul_1:z:0#ConB2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		*
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
:?????????		2
ConB2/BiasAddr

ConB2/ReluReluConB2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????		2

ConB2/Relu?
ConA2/Conv2D/ReadVariableOpReadVariableOp$cona2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConA2/Conv2D/ReadVariableOp?
ConA2/Conv2DConv2Ddropout/dropout/Mul_1:z:0#ConA2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
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
:?????????

2
ConA2/BiasAddr

ConA2/ReluReluConA2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

2

ConA2/Relu?
PoolB2/MaxPoolMaxPoolConB2/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
PoolB2/MaxPool?
PoolA2/MaxPoolMaxPoolConA2/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
PoolA2/MaxPool?
ConE/Conv2D/ReadVariableOpReadVariableOp#cone_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConE/Conv2D/ReadVariableOp?
ConE/Conv2DConv2Dinputs"ConE/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
ConE/Conv2D?
ConE/BiasAdd/ReadVariableOpReadVariableOp$cone_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConE/BiasAdd/ReadVariableOp?
ConE/BiasAddBiasAddConE/Conv2D:output:0#ConE/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
ConE/BiasAddo
	ConE/ReluReluConE/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
	ConE/Relu?
ConD/Conv2D/ReadVariableOpReadVariableOp#cond_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
ConD/Conv2D/ReadVariableOp?
ConD/Conv2DConv2Dinputs"ConD/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
ConD/Conv2D?
ConD/BiasAdd/ReadVariableOpReadVariableOp$cond_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConD/BiasAdd/ReadVariableOp?
ConD/BiasAddBiasAddConD/Conv2D:output:0#ConD/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
ConD/BiasAddo
	ConD/ReluReluConD/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
	ConD/Relu?
ConC/Conv2D/ReadVariableOpReadVariableOp#conc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConC/Conv2D/ReadVariableOp?
ConC/Conv2DConv2Dinputs"ConC/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
ConC/Conv2D?
ConC/BiasAdd/ReadVariableOpReadVariableOp$conc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConC/BiasAdd/ReadVariableOp?
ConC/BiasAddBiasAddConC/Conv2D:output:0#ConC/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
ConC/BiasAddo
	ConC/ReluReluConC/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
	ConC/Relu?
ConB3/Conv2D/ReadVariableOpReadVariableOp$conb3_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
ConB3/Conv2D/ReadVariableOp?
ConB3/Conv2DConv2DPoolB2/MaxPool:output:0#ConB3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2
ConB3/BiasAddr

ConB3/ReluReluConB3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(2

ConB3/Relu?
ConA3/Conv2D/ReadVariableOpReadVariableOp$cona3_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
ConA3/Conv2D/ReadVariableOp?
ConA3/Conv2DConv2DPoolA2/MaxPool:output:0#ConA3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2
ConA3/BiasAddr

ConA3/ReluReluConA3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(2

ConA3/Relu?
PoolE/MaxPoolMaxPoolConE/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
PoolE/MaxPool?
PoolD/MaxPoolMaxPoolConD/Relu:activations:0*/
_output_shapes
:?????????		*
ksize
*
paddingVALID*
strides
2
PoolD/MaxPool?
PoolC/MaxPoolMaxPoolConC/Relu:activations:0*/
_output_shapes
:?????????

*
ksize
*
paddingVALID*
strides
2
PoolC/MaxPool?
PoolB3/MaxPoolMaxPoolConB3/Relu:activations:0*/
_output_shapes
:?????????(*
ksize
*
paddingVALID*
strides
2
PoolB3/MaxPool?
PoolA3/MaxPoolMaxPoolConA3/Relu:activations:0*/
_output_shapes
:?????????(*
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
 *?8??2
dropout_4/dropout/Const?
dropout_4/dropout/MulMulPoolE/MaxPool:output:0 dropout_4/dropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout_4/dropout/Mulx
dropout_4/dropout/ShapeShapePoolE/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_4/dropout/Shape?
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
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
:?????????2 
dropout_4/dropout/GreaterEqual?
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout_4/dropout/Cast?
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
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
:?????????		2
dropout_3/dropout/Mulx
dropout_3/dropout/ShapeShapePoolD/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shape?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????		*
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
:?????????		2 
dropout_3/dropout/GreaterEqual?
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????		2
dropout_3/dropout/Cast?
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????		2
dropout_3/dropout/Mul_1w
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
:?????????

2
dropout_2/dropout/Mulx
dropout_2/dropout/ShapeShapePoolC/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????

*
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
:?????????

2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????

2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????

2
dropout_2/dropout/Mul_1w
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_8/dropout/Const?
dropout_8/dropout/MulMulPoolB3/MaxPool:output:0 dropout_8/dropout/Const:output:0*
T0*/
_output_shapes
:?????????(2
dropout_8/dropout/Muly
dropout_8/dropout/ShapeShapePoolB3/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_8/dropout/Shape?
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2 
dropout_8/dropout/GreaterEqual?
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????(2
dropout_8/dropout/Cast?
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????(2
dropout_8/dropout/Mul_1w
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_7/dropout/Const?
dropout_7/dropout/MulMulPoolB2/MaxPool:output:0 dropout_7/dropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout_7/dropout/Muly
dropout_7/dropout/ShapeShapePoolB2/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_7/dropout/Shape?
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
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
:?????????2 
dropout_7/dropout/GreaterEqual?
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout_7/dropout/Cast?
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
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
:?????????(2
dropout_6/dropout/Muly
dropout_6/dropout/ShapeShapePoolA3/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_6/dropout/Shape?
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2 
dropout_6/dropout/GreaterEqual?
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????(2
dropout_6/dropout/Cast?
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????(2
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
:?????????2
dropout_5/dropout/Muly
dropout_5/dropout/ShapeShapePoolA2/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_5/dropout/Shape?
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
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
:?????????2 
dropout_5/dropout/GreaterEqual?
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout_5/dropout/Cast?
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout_5/dropout/Mul_1s
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????`  2
flatten_8/Const?
flatten_8/ReshapeReshapedropout_4/dropout/Mul_1:z:0flatten_8/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_8/Reshapes
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_7/Const?
flatten_7/ReshapeReshapedropout_3/dropout/Mul_1:z:0flatten_7/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_7/Reshapes
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_6/Const?
flatten_6/ReshapeReshapedropout_2/dropout/Mul_1:z:0flatten_6/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_6/Reshapes
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????(   2
flatten_5/Const?
flatten_5/ReshapeReshapedropout_8/dropout/Mul_1:z:0flatten_5/Const:output:0*
T0*'
_output_shapes
:?????????(2
flatten_5/Reshapes
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
flatten_4/Const?
flatten_4/ReshapeReshapedropout_7/dropout/Mul_1:z:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_4/Reshapes
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_3/Const?
flatten_3/ReshapeReshapePoolB/MaxPool:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_3/Reshapes
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????(   2
flatten_2/Const?
flatten_2/ReshapeReshapedropout_6/dropout/Mul_1:z:0flatten_2/Const:output:0*
T0*'
_output_shapes
:?????????(2
flatten_2/Reshapes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_1/Const?
flatten_1/ReshapeReshapedropout_5/dropout/Mul_1:z:0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_1/Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????`  2
flatten/Const?
flatten/ReshapeReshapePoolA/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
HidA/MatMul/ReadVariableOpReadVariableOp#hida_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
HidA/MatMul/ReadVariableOp?
HidA/MatMulMatMulflatten/Reshape:output:0"HidA/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA/MatMul?
HidA/BiasAdd/ReadVariableOpReadVariableOp$hida_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidA/BiasAdd/ReadVariableOp?
HidA/BiasAddBiasAddHidA/MatMul:product:0#HidA/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA/BiasAddh
	HidA/ReluReluHidA/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
	HidA/Relu?
HidA2/MatMul/ReadVariableOpReadVariableOp$hida2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
HidA2/MatMul/ReadVariableOp?
HidA2/MatMulMatMulflatten_1/Reshape:output:0#HidA2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA2/MatMul?
HidA2/BiasAdd/ReadVariableOpReadVariableOp%hida2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidA2/BiasAdd/ReadVariableOp?
HidA2/BiasAddBiasAddHidA2/MatMul:product:0$HidA2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA2/BiasAddk

HidA2/ReluReluHidA2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidA2/Relu?
HidA3/MatMul/ReadVariableOpReadVariableOp$hida3_matmul_readvariableop_resource*
_output_shapes
:	(?*
dtype02
HidA3/MatMul/ReadVariableOp?
HidA3/MatMulMatMulflatten_2/Reshape:output:0#HidA3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA3/MatMul?
HidA3/BiasAdd/ReadVariableOpReadVariableOp%hida3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidA3/BiasAdd/ReadVariableOp?
HidA3/BiasAddBiasAddHidA3/MatMul:product:0$HidA3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA3/BiasAddk

HidA3/ReluReluHidA3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidA3/Relu?
HidB/MatMul/ReadVariableOpReadVariableOp#hidb_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
HidB/MatMul/ReadVariableOp?
HidB/MatMulMatMulflatten_3/Reshape:output:0"HidB/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidB/MatMul?
HidB/BiasAdd/ReadVariableOpReadVariableOp$hidb_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidB/BiasAdd/ReadVariableOp?
HidB/BiasAddBiasAddHidB/MatMul:product:0#HidB/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidB/BiasAddh
	HidB/ReluReluHidB/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
	HidB/Relu?
HidB2/MatMul/ReadVariableOpReadVariableOp$hidb2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
HidB2/MatMul/ReadVariableOp?
HidB2/MatMulMatMulflatten_4/Reshape:output:0#HidB2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidB2/MatMul?
HidB2/BiasAdd/ReadVariableOpReadVariableOp%hidb2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidB2/BiasAdd/ReadVariableOp?
HidB2/BiasAddBiasAddHidB2/MatMul:product:0$HidB2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidB2/BiasAddk

HidB2/ReluReluHidB2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidB2/Relu?
HidB3/MatMul/ReadVariableOpReadVariableOp$hidb3_matmul_readvariableop_resource*
_output_shapes
:	(?*
dtype02
HidB3/MatMul/ReadVariableOp?
HidB3/MatMulMatMulflatten_5/Reshape:output:0#HidB3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidB3/MatMul?
HidB3/BiasAdd/ReadVariableOpReadVariableOp%hidb3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidB3/BiasAdd/ReadVariableOp?
HidB3/BiasAddBiasAddHidB3/MatMul:product:0$HidB3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidB3/BiasAddk

HidB3/ReluReluHidB3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidB3/Relu?
HidC/MatMul/ReadVariableOpReadVariableOp#hidc_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
HidC/MatMul/ReadVariableOp?
HidC/MatMulMatMulflatten_6/Reshape:output:0"HidC/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidC/MatMul?
HidC/BiasAdd/ReadVariableOpReadVariableOp$hidc_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidC/BiasAdd/ReadVariableOp?
HidC/BiasAddBiasAddHidC/MatMul:product:0#HidC/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidC/BiasAddh
	HidC/ReluReluHidC/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
	HidC/Relu?
HidD1/MatMul/ReadVariableOpReadVariableOp$hidd1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
HidD1/MatMul/ReadVariableOp?
HidD1/MatMulMatMulflatten_7/Reshape:output:0#HidD1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidD1/MatMul?
HidD1/BiasAdd/ReadVariableOpReadVariableOp%hidd1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidD1/BiasAdd/ReadVariableOp?
HidD1/BiasAddBiasAddHidD1/MatMul:product:0$HidD1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidD1/BiasAddk

HidD1/ReluReluHidD1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidD1/Relu?
HidE1/MatMul/ReadVariableOpReadVariableOp$hide1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
HidE1/MatMul/ReadVariableOp?
HidE1/MatMulMatMulflatten_8/Reshape:output:0#HidE1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidE1/MatMul?
HidE1/BiasAdd/ReadVariableOpReadVariableOp%hide1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidE1/BiasAdd/ReadVariableOp?
HidE1/BiasAddBiasAddHidE1/MatMul:product:0$HidE1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidE1/BiasAddk

HidE1/ReluReluHidE1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidE1/Relut
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2HidA/Relu:activations:0HidA2/Relu:activations:0HidA3/Relu:activations:0HidB/Relu:activations:0HidB2/Relu:activations:0HidB3/Relu:activations:0HidC/Relu:activations:0HidD1/Relu:activations:0HidE1/Relu:activations:0 concatenate/concat/axis:output:0*
N	*
T0*(
_output_shapes
:??????????$2
concatenate/concat?
Hid1/MatMul/ReadVariableOpReadVariableOp#hid1_matmul_readvariableop_resource* 
_output_shapes
:
?$?*
dtype02
Hid1/MatMul/ReadVariableOp?
Hid1/MatMulMatMulconcatenate/concat:output:0"Hid1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Hid1/MatMul?
Hid1/BiasAdd/ReadVariableOpReadVariableOp$hid1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
Hid1/BiasAdd/ReadVariableOp?
Hid1/BiasAddBiasAddHid1/MatMul:product:0#Hid1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Hid1/BiasAddh
	Hid1/ReluReluHid1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
	Hid1/Relu?
Hid2/MatMul/ReadVariableOpReadVariableOp#hid2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Hid2/MatMul/ReadVariableOp?
Hid2/MatMulMatMulHid1/Relu:activations:0"Hid2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Hid2/MatMul?
Hid2/BiasAdd/ReadVariableOpReadVariableOp$hid2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
Hid2/BiasAdd/ReadVariableOp?
Hid2/BiasAddBiasAddHid2/MatMul:product:0#Hid2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Hid2/BiasAddh
	Hid2/ReluReluHid2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
	Hid2/Relu?
Hid3/MatMul/ReadVariableOpReadVariableOp#hid3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Hid3/MatMul/ReadVariableOp?
Hid3/MatMulMatMulHid2/Relu:activations:0"Hid3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Hid3/MatMul?
Hid3/BiasAdd/ReadVariableOpReadVariableOp$hid3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
Hid3/BiasAdd/ReadVariableOp?
Hid3/BiasAddBiasAddHid3/MatMul:product:0#Hid3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Hid3/BiasAddh
	Hid3/TanhTanhHid3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
	Hid3/Tanh?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMulHid3/Tanh:y:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
output/Softmaxs
IdentityIdentityoutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?

NoOpNoOp^ConA/BiasAdd/ReadVariableOp^ConA/Conv2D/ReadVariableOp^ConA2/BiasAdd/ReadVariableOp^ConA2/Conv2D/ReadVariableOp^ConA3/BiasAdd/ReadVariableOp^ConA3/Conv2D/ReadVariableOp^ConB/BiasAdd/ReadVariableOp^ConB/Conv2D/ReadVariableOp^ConB2/BiasAdd/ReadVariableOp^ConB2/Conv2D/ReadVariableOp^ConB3/BiasAdd/ReadVariableOp^ConB3/Conv2D/ReadVariableOp^ConC/BiasAdd/ReadVariableOp^ConC/Conv2D/ReadVariableOp^ConD/BiasAdd/ReadVariableOp^ConD/Conv2D/ReadVariableOp^ConE/BiasAdd/ReadVariableOp^ConE/Conv2D/ReadVariableOp^Hid1/BiasAdd/ReadVariableOp^Hid1/MatMul/ReadVariableOp^Hid2/BiasAdd/ReadVariableOp^Hid2/MatMul/ReadVariableOp^Hid3/BiasAdd/ReadVariableOp^Hid3/MatMul/ReadVariableOp^HidA/BiasAdd/ReadVariableOp^HidA/MatMul/ReadVariableOp^HidA2/BiasAdd/ReadVariableOp^HidA2/MatMul/ReadVariableOp^HidA3/BiasAdd/ReadVariableOp^HidA3/MatMul/ReadVariableOp^HidB/BiasAdd/ReadVariableOp^HidB/MatMul/ReadVariableOp^HidB2/BiasAdd/ReadVariableOp^HidB2/MatMul/ReadVariableOp^HidB3/BiasAdd/ReadVariableOp^HidB3/MatMul/ReadVariableOp^HidC/BiasAdd/ReadVariableOp^HidC/MatMul/ReadVariableOp^HidD1/BiasAdd/ReadVariableOp^HidD1/MatMul/ReadVariableOp^HidE1/BiasAdd/ReadVariableOp^HidE1/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
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
ConC/Conv2D/ReadVariableOpConC/Conv2D/ReadVariableOp2:
ConD/BiasAdd/ReadVariableOpConD/BiasAdd/ReadVariableOp28
ConD/Conv2D/ReadVariableOpConD/Conv2D/ReadVariableOp2:
ConE/BiasAdd/ReadVariableOpConE/BiasAdd/ReadVariableOp28
ConE/Conv2D/ReadVariableOpConE/Conv2D/ReadVariableOp2:
Hid1/BiasAdd/ReadVariableOpHid1/BiasAdd/ReadVariableOp28
Hid1/MatMul/ReadVariableOpHid1/MatMul/ReadVariableOp2:
Hid2/BiasAdd/ReadVariableOpHid2/BiasAdd/ReadVariableOp28
Hid2/MatMul/ReadVariableOpHid2/MatMul/ReadVariableOp2:
Hid3/BiasAdd/ReadVariableOpHid3/BiasAdd/ReadVariableOp28
Hid3/MatMul/ReadVariableOpHid3/MatMul/ReadVariableOp2:
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
HidD1/BiasAdd/ReadVariableOpHidD1/BiasAdd/ReadVariableOp2:
HidD1/MatMul/ReadVariableOpHidD1/MatMul/ReadVariableOp2<
HidE1/BiasAdd/ReadVariableOpHidE1/BiasAdd/ReadVariableOp2:
HidE1/MatMul/ReadVariableOpHidE1/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
?__inference_ConD_layer_call_and_return_conditional_losses_23480

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_7_layer_call_and_return_conditional_losses_21463

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
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
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
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_20761

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????		2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????		2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????		:W S
/
_output_shapes
:?????????		
 
_user_specified_nameinputs
?
?
@__inference_ConA3_layer_call_and_return_conditional_losses_23420

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
:?????????(*
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
:?????????(2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????(2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
]
A__inference_PoolA3_layer_call_and_return_conditional_losses_23520

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????(*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
@__inference_HidA2_layer_call_and_return_conditional_losses_23928

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
\
@__inference_PoolC_layer_call_and_return_conditional_losses_20464

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
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
`
D__inference_flatten_6_layer_call_and_return_conditional_losses_23866

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
c
D__inference_dropout_3_layer_call_and_return_conditional_losses_21532

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
:?????????		2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????		*
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
:?????????		2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????		2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????		2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????		2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????		:W S
/
_output_shapes
:?????????		
 
_user_specified_nameinputs
?
B
&__inference_PoolB3_layer_call_fn_23530

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
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolB3_layer_call_and_return_conditional_losses_207412
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
E
)__inference_dropout_8_layer_call_fn_23686

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
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_207752
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_20586

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
@__inference_HidB2_layer_call_and_return_conditional_losses_23988

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
$__inference_ConE_layer_call_fn_23489

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_ConE_layer_call_and_return_conditional_losses_206452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
]
A__inference_PoolA3_layer_call_and_return_conditional_losses_20420

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
A__inference_PoolB2_layer_call_and_return_conditional_losses_23395

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
B
&__inference_PoolB3_layer_call_fn_23525

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
A__inference_PoolB3_layer_call_and_return_conditional_losses_204422
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
`
D__inference_flatten_7_layer_call_and_return_conditional_losses_23877

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????		:W S
/
_output_shapes
:?????????		
 
_user_specified_nameinputs
?
E
)__inference_dropout_1_layer_call_fn_23298

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_205792
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
@__inference_HidB3_layer_call_and_return_conditional_losses_24008

inputs1
matmul_readvariableop_resource:	(?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(?*
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
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
c
D__inference_dropout_5_layer_call_and_return_conditional_losses_21417

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
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
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
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_6_layer_call_and_return_conditional_losses_23642

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????(2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????(2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
`
D__inference_flatten_8_layer_call_and_return_conditional_losses_23888

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????`  2
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
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
?__inference_ConD_layer_call_and_return_conditional_losses_20662

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_7_layer_call_and_return_conditional_losses_20812

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????		:W S
/
_output_shapes
:?????????		
 
_user_specified_nameinputs
?
C
'__inference_dropout_layer_call_fn_23271

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_205862
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_1_layer_call_fn_23303

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_217062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
@__inference_ConA2_layer_call_and_return_conditional_losses_20616

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
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
:?????????

2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????

2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????

2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_5_layer_call_and_return_conditional_losses_20828

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????(   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????(2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
\
@__inference_PoolA_layer_call_and_return_conditional_losses_23241

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
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
?__inference_ConA_layer_call_and_return_conditional_losses_20556

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_HidD1_layer_call_fn_24037

inputs
unknown:
??
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
@__inference_HidD1_layer_call_and_return_conditional_losses_210002
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
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
@__inference_HidD1_layer_call_and_return_conditional_losses_24048

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
A
%__inference_PoolE_layer_call_fn_23585

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
GPU2*0J 8? *I
fDRB
@__inference_PoolE_layer_call_and_return_conditional_losses_205082
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
D__inference_dropout_1_layer_call_and_return_conditional_losses_20579

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_5_layer_call_fn_23849

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_208282
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_21706

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
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
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
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
\
@__inference_PoolB_layer_call_and_return_conditional_losses_20566

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
@__inference_ConB2_layer_call_and_return_conditional_losses_23360

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		*
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
:?????????		2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????		2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????		2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$__inference_Hid2_layer_call_fn_24124

inputs
unknown:
??
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
GPU2*0J 8? *H
fCRA
?__inference_Hid2_layer_call_and_return_conditional_losses_210672
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
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?U
!__inference__traced_restore_25054
file_prefix6
assignvariableop_cona_kernel:*
assignvariableop_1_cona_bias:8
assignvariableop_2_conb_kernel:*
assignvariableop_3_conb_bias:9
assignvariableop_4_cona2_kernel:+
assignvariableop_5_cona2_bias:9
assignvariableop_6_conb2_kernel:+
assignvariableop_7_conb2_bias:9
assignvariableop_8_cona3_kernel:(+
assignvariableop_9_cona3_bias:(:
 assignvariableop_10_conb3_kernel:(,
assignvariableop_11_conb3_bias:(9
assignvariableop_12_conc_kernel:+
assignvariableop_13_conc_bias:9
assignvariableop_14_cond_kernel:		+
assignvariableop_15_cond_bias:9
assignvariableop_16_cone_kernel:+
assignvariableop_17_cone_bias:3
assignvariableop_18_hida_kernel:
??,
assignvariableop_19_hida_bias:	?4
 assignvariableop_20_hida2_kernel:
??-
assignvariableop_21_hida2_bias:	?3
 assignvariableop_22_hida3_kernel:	(?-
assignvariableop_23_hida3_bias:	?3
assignvariableop_24_hidb_kernel:
??,
assignvariableop_25_hidb_bias:	?4
 assignvariableop_26_hidb2_kernel:
??-
assignvariableop_27_hidb2_bias:	?3
 assignvariableop_28_hidb3_kernel:	(?-
assignvariableop_29_hidb3_bias:	?3
assignvariableop_30_hidc_kernel:
??,
assignvariableop_31_hidc_bias:	?4
 assignvariableop_32_hidd1_kernel:
??-
assignvariableop_33_hidd1_bias:	?4
 assignvariableop_34_hide1_kernel:
??-
assignvariableop_35_hide1_bias:	?3
assignvariableop_36_hid1_kernel:
?$?,
assignvariableop_37_hid1_bias:	?3
assignvariableop_38_hid2_kernel:
??,
assignvariableop_39_hid2_bias:	?3
assignvariableop_40_hid3_kernel:
??,
assignvariableop_41_hid3_bias:	?4
!assignvariableop_42_output_kernel:	?
-
assignvariableop_43_output_bias:
'
assignvariableop_44_adam_iter:	 )
assignvariableop_45_adam_beta_1: )
assignvariableop_46_adam_beta_2: (
assignvariableop_47_adam_decay: 0
&assignvariableop_48_adam_learning_rate: #
assignvariableop_49_total: #
assignvariableop_50_count: %
assignvariableop_51_total_1: %
assignvariableop_52_count_1: @
&assignvariableop_53_adam_cona_kernel_m:2
$assignvariableop_54_adam_cona_bias_m:@
&assignvariableop_55_adam_conb_kernel_m:2
$assignvariableop_56_adam_conb_bias_m:A
'assignvariableop_57_adam_cona2_kernel_m:3
%assignvariableop_58_adam_cona2_bias_m:A
'assignvariableop_59_adam_conb2_kernel_m:3
%assignvariableop_60_adam_conb2_bias_m:A
'assignvariableop_61_adam_cona3_kernel_m:(3
%assignvariableop_62_adam_cona3_bias_m:(A
'assignvariableop_63_adam_conb3_kernel_m:(3
%assignvariableop_64_adam_conb3_bias_m:(@
&assignvariableop_65_adam_conc_kernel_m:2
$assignvariableop_66_adam_conc_bias_m:@
&assignvariableop_67_adam_cond_kernel_m:		2
$assignvariableop_68_adam_cond_bias_m:@
&assignvariableop_69_adam_cone_kernel_m:2
$assignvariableop_70_adam_cone_bias_m::
&assignvariableop_71_adam_hida_kernel_m:
??3
$assignvariableop_72_adam_hida_bias_m:	?;
'assignvariableop_73_adam_hida2_kernel_m:
??4
%assignvariableop_74_adam_hida2_bias_m:	?:
'assignvariableop_75_adam_hida3_kernel_m:	(?4
%assignvariableop_76_adam_hida3_bias_m:	?:
&assignvariableop_77_adam_hidb_kernel_m:
??3
$assignvariableop_78_adam_hidb_bias_m:	?;
'assignvariableop_79_adam_hidb2_kernel_m:
??4
%assignvariableop_80_adam_hidb2_bias_m:	?:
'assignvariableop_81_adam_hidb3_kernel_m:	(?4
%assignvariableop_82_adam_hidb3_bias_m:	?:
&assignvariableop_83_adam_hidc_kernel_m:
??3
$assignvariableop_84_adam_hidc_bias_m:	?;
'assignvariableop_85_adam_hidd1_kernel_m:
??4
%assignvariableop_86_adam_hidd1_bias_m:	?;
'assignvariableop_87_adam_hide1_kernel_m:
??4
%assignvariableop_88_adam_hide1_bias_m:	?:
&assignvariableop_89_adam_hid1_kernel_m:
?$?3
$assignvariableop_90_adam_hid1_bias_m:	?:
&assignvariableop_91_adam_hid2_kernel_m:
??3
$assignvariableop_92_adam_hid2_bias_m:	?:
&assignvariableop_93_adam_hid3_kernel_m:
??3
$assignvariableop_94_adam_hid3_bias_m:	?;
(assignvariableop_95_adam_output_kernel_m:	?
4
&assignvariableop_96_adam_output_bias_m:
@
&assignvariableop_97_adam_cona_kernel_v:2
$assignvariableop_98_adam_cona_bias_v:@
&assignvariableop_99_adam_conb_kernel_v:3
%assignvariableop_100_adam_conb_bias_v:B
(assignvariableop_101_adam_cona2_kernel_v:4
&assignvariableop_102_adam_cona2_bias_v:B
(assignvariableop_103_adam_conb2_kernel_v:4
&assignvariableop_104_adam_conb2_bias_v:B
(assignvariableop_105_adam_cona3_kernel_v:(4
&assignvariableop_106_adam_cona3_bias_v:(B
(assignvariableop_107_adam_conb3_kernel_v:(4
&assignvariableop_108_adam_conb3_bias_v:(A
'assignvariableop_109_adam_conc_kernel_v:3
%assignvariableop_110_adam_conc_bias_v:A
'assignvariableop_111_adam_cond_kernel_v:		3
%assignvariableop_112_adam_cond_bias_v:A
'assignvariableop_113_adam_cone_kernel_v:3
%assignvariableop_114_adam_cone_bias_v:;
'assignvariableop_115_adam_hida_kernel_v:
??4
%assignvariableop_116_adam_hida_bias_v:	?<
(assignvariableop_117_adam_hida2_kernel_v:
??5
&assignvariableop_118_adam_hida2_bias_v:	?;
(assignvariableop_119_adam_hida3_kernel_v:	(?5
&assignvariableop_120_adam_hida3_bias_v:	?;
'assignvariableop_121_adam_hidb_kernel_v:
??4
%assignvariableop_122_adam_hidb_bias_v:	?<
(assignvariableop_123_adam_hidb2_kernel_v:
??5
&assignvariableop_124_adam_hidb2_bias_v:	?;
(assignvariableop_125_adam_hidb3_kernel_v:	(?5
&assignvariableop_126_adam_hidb3_bias_v:	?;
'assignvariableop_127_adam_hidc_kernel_v:
??4
%assignvariableop_128_adam_hidc_bias_v:	?<
(assignvariableop_129_adam_hidd1_kernel_v:
??5
&assignvariableop_130_adam_hidd1_bias_v:	?<
(assignvariableop_131_adam_hide1_kernel_v:
??5
&assignvariableop_132_adam_hide1_bias_v:	?;
'assignvariableop_133_adam_hid1_kernel_v:
?$?4
%assignvariableop_134_adam_hid1_bias_v:	?;
'assignvariableop_135_adam_hid2_kernel_v:
??4
%assignvariableop_136_adam_hid2_bias_v:	?;
'assignvariableop_137_adam_hid3_kernel_v:
??4
%assignvariableop_138_adam_hid3_bias_v:	?<
)assignvariableop_139_adam_output_kernel_v:	?
5
'assignvariableop_140_adam_output_bias_v:

identity_142??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_120?AssignVariableOp_121?AssignVariableOp_122?AssignVariableOp_123?AssignVariableOp_124?AssignVariableOp_125?AssignVariableOp_126?AssignVariableOp_127?AssignVariableOp_128?AssignVariableOp_129?AssignVariableOp_13?AssignVariableOp_130?AssignVariableOp_131?AssignVariableOp_132?AssignVariableOp_133?AssignVariableOp_134?AssignVariableOp_135?AssignVariableOp_136?AssignVariableOp_137?AssignVariableOp_138?AssignVariableOp_139?AssignVariableOp_14?AssignVariableOp_140?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?Q
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?P
value?PB?P?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
?2?	2
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
AssignVariableOp_6AssignVariableOpassignvariableop_6_conb2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_conb2_biasIdentity_7:output:0"/device:CPU:0*
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
AssignVariableOp_10AssignVariableOp assignvariableop_10_conb3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_conb3_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_conc_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_conc_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_cond_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_cond_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_cone_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_cone_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_hida_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_hida_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp assignvariableop_20_hida2_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_hida2_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp assignvariableop_22_hida3_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_hida3_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_hidb_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_hidb_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp assignvariableop_26_hidb2_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_hidb2_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp assignvariableop_28_hidb3_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_hidb3_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_hidc_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_hidc_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp assignvariableop_32_hidd1_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_hidd1_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp assignvariableop_34_hide1_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpassignvariableop_35_hide1_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpassignvariableop_36_hid1_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpassignvariableop_37_hid1_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpassignvariableop_38_hid2_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpassignvariableop_39_hid2_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpassignvariableop_40_hid3_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpassignvariableop_41_hid3_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp!assignvariableop_42_output_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpassignvariableop_43_output_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpassignvariableop_44_adam_iterIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpassignvariableop_45_adam_beta_1Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOpassignvariableop_46_adam_beta_2Identity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOpassignvariableop_47_adam_decayIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp&assignvariableop_48_adam_learning_rateIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOpassignvariableop_49_totalIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpassignvariableop_50_countIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOpassignvariableop_51_total_1Identity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOpassignvariableop_52_count_1Identity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp&assignvariableop_53_adam_cona_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp$assignvariableop_54_adam_cona_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp&assignvariableop_55_adam_conb_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp$assignvariableop_56_adam_conb_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp'assignvariableop_57_adam_cona2_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp%assignvariableop_58_adam_cona2_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp'assignvariableop_59_adam_conb2_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp%assignvariableop_60_adam_conb2_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp'assignvariableop_61_adam_cona3_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp%assignvariableop_62_adam_cona3_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp'assignvariableop_63_adam_conb3_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp%assignvariableop_64_adam_conb3_bias_mIdentity_64:output:0"/device:CPU:0*
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
AssignVariableOp_67AssignVariableOp&assignvariableop_67_adam_cond_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp$assignvariableop_68_adam_cond_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp&assignvariableop_69_adam_cone_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp$assignvariableop_70_adam_cone_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp&assignvariableop_71_adam_hida_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp$assignvariableop_72_adam_hida_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp'assignvariableop_73_adam_hida2_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp%assignvariableop_74_adam_hida2_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp'assignvariableop_75_adam_hida3_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp%assignvariableop_76_adam_hida3_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp&assignvariableop_77_adam_hidb_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp$assignvariableop_78_adam_hidb_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp'assignvariableop_79_adam_hidb2_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp%assignvariableop_80_adam_hidb2_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp'assignvariableop_81_adam_hidb3_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp%assignvariableop_82_adam_hidb3_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp&assignvariableop_83_adam_hidc_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp$assignvariableop_84_adam_hidc_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp'assignvariableop_85_adam_hidd1_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp%assignvariableop_86_adam_hidd1_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp'assignvariableop_87_adam_hide1_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp%assignvariableop_88_adam_hide1_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp&assignvariableop_89_adam_hid1_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp$assignvariableop_90_adam_hid1_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp&assignvariableop_91_adam_hid2_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp$assignvariableop_92_adam_hid2_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp&assignvariableop_93_adam_hid3_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp$assignvariableop_94_adam_hid3_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp(assignvariableop_95_adam_output_kernel_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp&assignvariableop_96_adam_output_bias_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOp&assignvariableop_97_adam_cona_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOp$assignvariableop_98_adam_cona_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOp&assignvariableop_99_adam_conb_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOp%assignvariableop_100_adam_conb_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOp(assignvariableop_101_adam_cona2_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOp&assignvariableop_102_adam_cona2_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOp(assignvariableop_103_adam_conb2_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104?
AssignVariableOp_104AssignVariableOp&assignvariableop_104_adam_conb2_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105?
AssignVariableOp_105AssignVariableOp(assignvariableop_105_adam_cona3_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106?
AssignVariableOp_106AssignVariableOp&assignvariableop_106_adam_cona3_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107?
AssignVariableOp_107AssignVariableOp(assignvariableop_107_adam_conb3_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108?
AssignVariableOp_108AssignVariableOp&assignvariableop_108_adam_conb3_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109?
AssignVariableOp_109AssignVariableOp'assignvariableop_109_adam_conc_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110?
AssignVariableOp_110AssignVariableOp%assignvariableop_110_adam_conc_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111?
AssignVariableOp_111AssignVariableOp'assignvariableop_111_adam_cond_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112?
AssignVariableOp_112AssignVariableOp%assignvariableop_112_adam_cond_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113?
AssignVariableOp_113AssignVariableOp'assignvariableop_113_adam_cone_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114?
AssignVariableOp_114AssignVariableOp%assignvariableop_114_adam_cone_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115?
AssignVariableOp_115AssignVariableOp'assignvariableop_115_adam_hida_kernel_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116?
AssignVariableOp_116AssignVariableOp%assignvariableop_116_adam_hida_bias_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117?
AssignVariableOp_117AssignVariableOp(assignvariableop_117_adam_hida2_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118?
AssignVariableOp_118AssignVariableOp&assignvariableop_118_adam_hida2_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119?
AssignVariableOp_119AssignVariableOp(assignvariableop_119_adam_hida3_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120?
AssignVariableOp_120AssignVariableOp&assignvariableop_120_adam_hida3_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121?
AssignVariableOp_121AssignVariableOp'assignvariableop_121_adam_hidb_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122?
AssignVariableOp_122AssignVariableOp%assignvariableop_122_adam_hidb_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123?
AssignVariableOp_123AssignVariableOp(assignvariableop_123_adam_hidb2_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124?
AssignVariableOp_124AssignVariableOp&assignvariableop_124_adam_hidb2_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125?
AssignVariableOp_125AssignVariableOp(assignvariableop_125_adam_hidb3_kernel_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126?
AssignVariableOp_126AssignVariableOp&assignvariableop_126_adam_hidb3_bias_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127?
AssignVariableOp_127AssignVariableOp'assignvariableop_127_adam_hidc_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128?
AssignVariableOp_128AssignVariableOp%assignvariableop_128_adam_hidc_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129?
AssignVariableOp_129AssignVariableOp(assignvariableop_129_adam_hidd1_kernel_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130?
AssignVariableOp_130AssignVariableOp&assignvariableop_130_adam_hidd1_bias_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131?
AssignVariableOp_131AssignVariableOp(assignvariableop_131_adam_hide1_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132?
AssignVariableOp_132AssignVariableOp&assignvariableop_132_adam_hide1_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_132q
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:2
Identity_133?
AssignVariableOp_133AssignVariableOp'assignvariableop_133_adam_hid1_kernel_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_133q
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:2
Identity_134?
AssignVariableOp_134AssignVariableOp%assignvariableop_134_adam_hid1_bias_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_134q
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:2
Identity_135?
AssignVariableOp_135AssignVariableOp'assignvariableop_135_adam_hid2_kernel_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_135q
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:2
Identity_136?
AssignVariableOp_136AssignVariableOp%assignvariableop_136_adam_hid2_bias_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_136q
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:2
Identity_137?
AssignVariableOp_137AssignVariableOp'assignvariableop_137_adam_hid3_kernel_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_137q
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:2
Identity_138?
AssignVariableOp_138AssignVariableOp%assignvariableop_138_adam_hid3_bias_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_138q
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:2
Identity_139?
AssignVariableOp_139AssignVariableOp)assignvariableop_139_adam_output_kernel_vIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139q
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:2
Identity_140?
AssignVariableOp_140AssignVariableOp'assignvariableop_140_adam_output_bias_vIdentity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1409
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_141Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_141i
Identity_142IdentityIdentity_141:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_142?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"%
identity_142Identity_142:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_140AssignVariableOp_1402*
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
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_23308

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
?__inference_ConB_layer_call_and_return_conditional_losses_20539

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
]
A__inference_PoolB3_layer_call_and_return_conditional_losses_23540

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????(*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
]
A__inference_PoolA2_layer_call_and_return_conditional_losses_20376

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
b
D__inference_dropout_7_layer_call_and_return_conditional_losses_23669

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
\
@__inference_PoolC_layer_call_and_return_conditional_losses_23555

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
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
a
B__inference_dropout_layer_call_and_return_conditional_losses_21683

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
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
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
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
A
%__inference_PoolD_layer_call_fn_23570

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
:?????????		* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_PoolD_layer_call_and_return_conditional_losses_207292
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????		2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_output_layer_call_fn_24164

inputs
unknown:	?

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_211012
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

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
?
E
)__inference_dropout_7_layer_call_fn_23659

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_207822
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
?__inference_Hid2_layer_call_and_return_conditional_losses_21067

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_20796

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
\
@__inference_PoolB_layer_call_and_return_conditional_losses_23261

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
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
$__inference_Hid3_layer_call_fn_24144

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
GPU2*0J 8? *H
fCRA
?__inference_Hid3_layer_call_and_return_conditional_losses_210842
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
E
)__inference_dropout_2_layer_call_fn_23713

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
:?????????

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_207682
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????

2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_20852

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????(   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????(2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
%__inference_ConB3_layer_call_fn_23429

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
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConB3_layer_call_and_return_conditional_losses_206962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_4_layer_call_fn_23838

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
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_208362
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_8_layer_call_fn_23691

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
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_214862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
?__inference_ConE_layer_call_and_return_conditional_losses_23500

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
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
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$__inference_ConC_layer_call_fn_23449

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_ConC_layer_call_and_return_conditional_losses_206792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
?__inference_HidC_layer_call_and_return_conditional_losses_20983

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
@__inference_ConB3_layer_call_and_return_conditional_losses_20696

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
:?????????(*
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
:?????????(2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????(2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
?__inference_HidC_layer_call_and_return_conditional_losses_24028

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
A
%__inference_PoolA_layer_call_fn_23236

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_PoolA_layer_call_and_return_conditional_losses_205722
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_1_layer_call_fn_23805

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
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_208602
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_HidE1_layer_call_fn_24057

inputs
unknown:
??
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
@__inference_HidE1_layer_call_and_return_conditional_losses_210172
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
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_6_layer_call_and_return_conditional_losses_21440

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
:?????????(2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????(2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????(2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
B
&__inference_PoolB2_layer_call_fn_23390

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolB2_layer_call_and_return_conditional_losses_206262
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????		:W S
/
_output_shapes
:?????????		
 
_user_specified_nameinputs
?
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_23750

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????		2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????		2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????		:W S
/
_output_shapes
:?????????		
 
_user_specified_nameinputs
?
?
%__inference_HidB3_layer_call_fn_23997

inputs
unknown:	(?
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
@__inference_HidB3_layer_call_and_return_conditional_losses_209662
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
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_21199	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:		

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:(

unknown_14:($

unknown_15:(

unknown_16:(

unknown_17:
??

unknown_18:	?

unknown_19:
??

unknown_20:	?

unknown_21:	(?

unknown_22:	?

unknown_23:
??

unknown_24:	?

unknown_25:
??

unknown_26:	?

unknown_27:	(?

unknown_28:	?

unknown_29:
??

unknown_30:	?

unknown_31:
??

unknown_32:	?

unknown_33:
??

unknown_34:	?

unknown_35:
?$?

unknown_36:	?

unknown_37:
??

unknown_38:	?

unknown_39:
??

unknown_40:	?

unknown_41:	?


unknown_42:

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
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_211082
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?
?
%__inference_HidA2_layer_call_fn_23917

inputs
unknown:
??
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
@__inference_HidA2_layer_call_and_return_conditional_losses_208982
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
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
A
%__inference_PoolA_layer_call_fn_23231

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
GPU2*0J 8? *I
fDRB
@__inference_PoolA_layer_call_and_return_conditional_losses_203322
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
]
A__inference_PoolA2_layer_call_and_return_conditional_losses_23380

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
\
@__inference_PoolC_layer_call_and_return_conditional_losses_20735

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????

*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????

2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_6_layer_call_and_return_conditional_losses_23654

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
:?????????(2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????(2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????(2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_20860

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_ConA3_layer_call_fn_23409

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
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConA3_layer_call_and_return_conditional_losses_207132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_23293

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
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
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
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
\
@__inference_PoolD_layer_call_and_return_conditional_losses_23575

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
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
\
@__inference_PoolD_layer_call_and_return_conditional_losses_23580

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????		*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????		2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_6_layer_call_fn_23860

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
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_208202
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
A
%__inference_PoolC_layer_call_fn_23545

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
GPU2*0J 8? *I
fDRB
@__inference_PoolC_layer_call_and_return_conditional_losses_204642
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
D__inference_dropout_8_layer_call_and_return_conditional_losses_23696

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????(2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????(2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
`
D__inference_flatten_4_layer_call_and_return_conditional_losses_20836

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
@__inference_ConA3_layer_call_and_return_conditional_losses_20713

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
:?????????(*
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
:?????????(2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????(2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
\
@__inference_PoolA_layer_call_and_return_conditional_losses_20332

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
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
\
@__inference_PoolA_layer_call_and_return_conditional_losses_20572

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_2_layer_call_fn_23816

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_208522
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
$__inference_ConA_layer_call_fn_23195

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_ConA_layer_call_and_return_conditional_losses_205562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_23833

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_21509

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
:?????????

2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????

*
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
:?????????

2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????

2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????

2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????

2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
A
%__inference_PoolC_layer_call_fn_23550

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
:?????????

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_PoolC_layer_call_and_return_conditional_losses_207352
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????

2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_5_layer_call_fn_23605

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_207962
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$__inference_HidC_layer_call_fn_24017

inputs
unknown:
??
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
GPU2*0J 8? *H
fCRA
?__inference_HidC_layer_call_and_return_conditional_losses_209832
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
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_8_layer_call_and_return_conditional_losses_23708

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
:?????????(2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????(2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????(2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
E
)__inference_flatten_8_layer_call_fn_23882

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
GPU2*0J 8? *M
fHRF
D__inference_flatten_8_layer_call_and_return_conditional_losses_208042
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$__inference_HidB_layer_call_fn_23957

inputs
unknown:
??
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
GPU2*0J 8? *H
fCRA
?__inference_HidB_layer_call_and_return_conditional_losses_209322
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
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
C
'__inference_flatten_layer_call_fn_23794

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
GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_208682
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_6_layer_call_fn_23637

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
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_214402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
`
D__inference_flatten_6_layer_call_and_return_conditional_losses_20820

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
]
A__inference_PoolB2_layer_call_and_return_conditional_losses_23400

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????		:W S
/
_output_shapes
:?????????		
 
_user_specified_nameinputs
?
b
D__inference_dropout_7_layer_call_and_return_conditional_losses_20782

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
A
%__inference_PoolB_layer_call_fn_23251

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
GPU2*0J 8? *I
fDRB
@__inference_PoolB_layer_call_and_return_conditional_losses_203542
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
B__inference_flatten_layer_call_and_return_conditional_losses_20868

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????`  2
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
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
B
&__inference_PoolA2_layer_call_fn_23370

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolA2_layer_call_and_return_conditional_losses_206322
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
b
D__inference_dropout_6_layer_call_and_return_conditional_losses_20789

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????(2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????(2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
b
D__inference_dropout_8_layer_call_and_return_conditional_losses_20775

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????(2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????(2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
]
A__inference_PoolB3_layer_call_and_return_conditional_losses_23535

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
B
&__inference_PoolB2_layer_call_fn_23385

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
A__inference_PoolB2_layer_call_and_return_conditional_losses_203982
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
$__inference_Hid1_layer_call_fn_24104

inputs
unknown:
?$?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_Hid1_layer_call_and_return_conditional_losses_210502
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
?
?
A__inference_output_layer_call_and_return_conditional_losses_24175

inputs1
matmul_readvariableop_resource:	?
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
2

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
?__inference_Hid1_layer_call_and_return_conditional_losses_24115

inputs2
matmul_readvariableop_resource:
?$?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?$?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
?
\
@__inference_PoolD_layer_call_and_return_conditional_losses_20729

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????		*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????		2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_23615

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$__inference_ConD_layer_call_fn_23469

inputs!
unknown:		
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_ConD_layer_call_and_return_conditional_losses_206622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
?__inference_HidA_layer_call_and_return_conditional_losses_20881

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
B
&__inference_PoolA3_layer_call_fn_23505

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
A__inference_PoolA3_layer_call_and_return_conditional_losses_204202
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
\
@__inference_PoolA_layer_call_and_return_conditional_losses_23246

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
?__inference_ConB_layer_call_and_return_conditional_losses_23226

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
B
&__inference_PoolA2_layer_call_fn_23365

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
A__inference_PoolA2_layer_call_and_return_conditional_losses_203762
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
]
A__inference_PoolA2_layer_call_and_return_conditional_losses_20632

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
\
@__inference_PoolC_layer_call_and_return_conditional_losses_23560

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????

*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????

2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_8_layer_call_and_return_conditional_losses_20804

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????`  2
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
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_20768

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????

2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????

2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
A
%__inference_PoolE_layer_call_fn_23590

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_PoolE_layer_call_and_return_conditional_losses_207232
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$__inference_ConB_layer_call_fn_23215

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_ConB_layer_call_and_return_conditional_losses_205392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
@__inference_HidD1_layer_call_and_return_conditional_losses_21000

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_21555

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
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
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
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_23800

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????`  2
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
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_concatenate_layer_call_fn_24081
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_210372
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/2:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/3:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/4:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/5:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/6:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/7:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/8
?
`
D__inference_flatten_5_layer_call_and_return_conditional_losses_23855

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????(   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????(2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
A
%__inference_PoolD_layer_call_fn_23565

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
GPU2*0J 8? *I
fDRB
@__inference_PoolD_layer_call_and_return_conditional_losses_204862
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
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_23811

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
?__inference_Hid2_layer_call_and_return_conditional_losses_24135

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
%__inference_HidB2_layer_call_fn_23977

inputs
unknown:
??
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
@__inference_HidB2_layer_call_and_return_conditional_losses_209492
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
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
?__inference_HidB_layer_call_and_return_conditional_losses_20932

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
?__inference_HidB_layer_call_and_return_conditional_losses_23968

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
?__inference_ConC_layer_call_and_return_conditional_losses_23460

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
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
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
]
A__inference_PoolB3_layer_call_and_return_conditional_losses_20741

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????(*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
A__inference_output_layer_call_and_return_conditional_losses_21101

inputs1
matmul_readvariableop_resource:	?
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
2

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
?
]
A__inference_PoolB3_layer_call_and_return_conditional_losses_20442

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
?__inference_HidA_layer_call_and_return_conditional_losses_23908

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_3_layer_call_fn_23745

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
:?????????		* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_215322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????		2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????		22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????		
 
_user_specified_nameinputs
??
?"
 __inference__wrapped_model_20323	
inputC
)model_conb_conv2d_readvariableop_resource:8
*model_conb_biasadd_readvariableop_resource:C
)model_cona_conv2d_readvariableop_resource:8
*model_cona_biasadd_readvariableop_resource:D
*model_conb2_conv2d_readvariableop_resource:9
+model_conb2_biasadd_readvariableop_resource:D
*model_cona2_conv2d_readvariableop_resource:9
+model_cona2_biasadd_readvariableop_resource:C
)model_cone_conv2d_readvariableop_resource:8
*model_cone_biasadd_readvariableop_resource:C
)model_cond_conv2d_readvariableop_resource:		8
*model_cond_biasadd_readvariableop_resource:C
)model_conc_conv2d_readvariableop_resource:8
*model_conc_biasadd_readvariableop_resource:D
*model_conb3_conv2d_readvariableop_resource:(9
+model_conb3_biasadd_readvariableop_resource:(D
*model_cona3_conv2d_readvariableop_resource:(9
+model_cona3_biasadd_readvariableop_resource:(=
)model_hida_matmul_readvariableop_resource:
??9
*model_hida_biasadd_readvariableop_resource:	?>
*model_hida2_matmul_readvariableop_resource:
??:
+model_hida2_biasadd_readvariableop_resource:	?=
*model_hida3_matmul_readvariableop_resource:	(?:
+model_hida3_biasadd_readvariableop_resource:	?=
)model_hidb_matmul_readvariableop_resource:
??9
*model_hidb_biasadd_readvariableop_resource:	?>
*model_hidb2_matmul_readvariableop_resource:
??:
+model_hidb2_biasadd_readvariableop_resource:	?=
*model_hidb3_matmul_readvariableop_resource:	(?:
+model_hidb3_biasadd_readvariableop_resource:	?=
)model_hidc_matmul_readvariableop_resource:
??9
*model_hidc_biasadd_readvariableop_resource:	?>
*model_hidd1_matmul_readvariableop_resource:
??:
+model_hidd1_biasadd_readvariableop_resource:	?>
*model_hide1_matmul_readvariableop_resource:
??:
+model_hide1_biasadd_readvariableop_resource:	?=
)model_hid1_matmul_readvariableop_resource:
?$?9
*model_hid1_biasadd_readvariableop_resource:	?=
)model_hid2_matmul_readvariableop_resource:
??9
*model_hid2_biasadd_readvariableop_resource:	?=
)model_hid3_matmul_readvariableop_resource:
??9
*model_hid3_biasadd_readvariableop_resource:	?>
+model_output_matmul_readvariableop_resource:	?
:
,model_output_biasadd_readvariableop_resource:

identity??!model/ConA/BiasAdd/ReadVariableOp? model/ConA/Conv2D/ReadVariableOp?"model/ConA2/BiasAdd/ReadVariableOp?!model/ConA2/Conv2D/ReadVariableOp?"model/ConA3/BiasAdd/ReadVariableOp?!model/ConA3/Conv2D/ReadVariableOp?!model/ConB/BiasAdd/ReadVariableOp? model/ConB/Conv2D/ReadVariableOp?"model/ConB2/BiasAdd/ReadVariableOp?!model/ConB2/Conv2D/ReadVariableOp?"model/ConB3/BiasAdd/ReadVariableOp?!model/ConB3/Conv2D/ReadVariableOp?!model/ConC/BiasAdd/ReadVariableOp? model/ConC/Conv2D/ReadVariableOp?!model/ConD/BiasAdd/ReadVariableOp? model/ConD/Conv2D/ReadVariableOp?!model/ConE/BiasAdd/ReadVariableOp? model/ConE/Conv2D/ReadVariableOp?!model/Hid1/BiasAdd/ReadVariableOp? model/Hid1/MatMul/ReadVariableOp?!model/Hid2/BiasAdd/ReadVariableOp? model/Hid2/MatMul/ReadVariableOp?!model/Hid3/BiasAdd/ReadVariableOp? model/Hid3/MatMul/ReadVariableOp?!model/HidA/BiasAdd/ReadVariableOp? model/HidA/MatMul/ReadVariableOp?"model/HidA2/BiasAdd/ReadVariableOp?!model/HidA2/MatMul/ReadVariableOp?"model/HidA3/BiasAdd/ReadVariableOp?!model/HidA3/MatMul/ReadVariableOp?!model/HidB/BiasAdd/ReadVariableOp? model/HidB/MatMul/ReadVariableOp?"model/HidB2/BiasAdd/ReadVariableOp?!model/HidB2/MatMul/ReadVariableOp?"model/HidB3/BiasAdd/ReadVariableOp?!model/HidB3/MatMul/ReadVariableOp?!model/HidC/BiasAdd/ReadVariableOp? model/HidC/MatMul/ReadVariableOp?"model/HidD1/BiasAdd/ReadVariableOp?!model/HidD1/MatMul/ReadVariableOp?"model/HidE1/BiasAdd/ReadVariableOp?!model/HidE1/MatMul/ReadVariableOp?#model/output/BiasAdd/ReadVariableOp?"model/output/MatMul/ReadVariableOp?
 model/ConB/Conv2D/ReadVariableOpReadVariableOp)model_conb_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 model/ConB/Conv2D/ReadVariableOp?
model/ConB/Conv2DConv2Dinput(model/ConB/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model/ConB/Conv2D?
!model/ConB/BiasAdd/ReadVariableOpReadVariableOp*model_conb_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!model/ConB/BiasAdd/ReadVariableOp?
model/ConB/BiasAddBiasAddmodel/ConB/Conv2D:output:0)model/ConB/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model/ConB/BiasAdd?
model/ConB/ReluRelumodel/ConB/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model/ConB/Relu?
 model/ConA/Conv2D/ReadVariableOpReadVariableOp)model_cona_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 model/ConA/Conv2D/ReadVariableOp?
model/ConA/Conv2DConv2Dinput(model/ConA/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model/ConA/Conv2D?
!model/ConA/BiasAdd/ReadVariableOpReadVariableOp*model_cona_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!model/ConA/BiasAdd/ReadVariableOp?
model/ConA/BiasAddBiasAddmodel/ConA/Conv2D:output:0)model/ConA/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model/ConA/BiasAdd?
model/ConA/ReluRelumodel/ConA/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model/ConA/Relu?
model/PoolB/MaxPoolMaxPoolmodel/ConB/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
model/PoolB/MaxPool?
model/PoolA/MaxPoolMaxPoolmodel/ConA/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
model/PoolA/MaxPool?
model/dropout_1/IdentityIdentitymodel/PoolB/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
model/dropout_1/Identity?
model/dropout/IdentityIdentitymodel/PoolA/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
model/dropout/Identity?
!model/ConB2/Conv2D/ReadVariableOpReadVariableOp*model_conb2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02#
!model/ConB2/Conv2D/ReadVariableOp?
model/ConB2/Conv2DConv2D!model/dropout_1/Identity:output:0)model/ConB2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		*
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
:?????????		2
model/ConB2/BiasAdd?
model/ConB2/ReluRelumodel/ConB2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????		2
model/ConB2/Relu?
!model/ConA2/Conv2D/ReadVariableOpReadVariableOp*model_cona2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02#
!model/ConA2/Conv2D/ReadVariableOp?
model/ConA2/Conv2DConv2Dmodel/dropout/Identity:output:0)model/ConA2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
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
:?????????

2
model/ConA2/BiasAdd?
model/ConA2/ReluRelumodel/ConA2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

2
model/ConA2/Relu?
model/PoolB2/MaxPoolMaxPoolmodel/ConB2/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
model/PoolB2/MaxPool?
model/PoolA2/MaxPoolMaxPoolmodel/ConA2/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
model/PoolA2/MaxPool?
 model/ConE/Conv2D/ReadVariableOpReadVariableOp)model_cone_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 model/ConE/Conv2D/ReadVariableOp?
model/ConE/Conv2DConv2Dinput(model/ConE/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model/ConE/Conv2D?
!model/ConE/BiasAdd/ReadVariableOpReadVariableOp*model_cone_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!model/ConE/BiasAdd/ReadVariableOp?
model/ConE/BiasAddBiasAddmodel/ConE/Conv2D:output:0)model/ConE/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model/ConE/BiasAdd?
model/ConE/ReluRelumodel/ConE/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model/ConE/Relu?
 model/ConD/Conv2D/ReadVariableOpReadVariableOp)model_cond_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02"
 model/ConD/Conv2D/ReadVariableOp?
model/ConD/Conv2DConv2Dinput(model/ConD/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model/ConD/Conv2D?
!model/ConD/BiasAdd/ReadVariableOpReadVariableOp*model_cond_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!model/ConD/BiasAdd/ReadVariableOp?
model/ConD/BiasAddBiasAddmodel/ConD/Conv2D:output:0)model/ConD/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model/ConD/BiasAdd?
model/ConD/ReluRelumodel/ConD/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model/ConD/Relu?
 model/ConC/Conv2D/ReadVariableOpReadVariableOp)model_conc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 model/ConC/Conv2D/ReadVariableOp?
model/ConC/Conv2DConv2Dinput(model/ConC/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model/ConC/Conv2D?
!model/ConC/BiasAdd/ReadVariableOpReadVariableOp*model_conc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!model/ConC/BiasAdd/ReadVariableOp?
model/ConC/BiasAddBiasAddmodel/ConC/Conv2D:output:0)model/ConC/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model/ConC/BiasAdd?
model/ConC/ReluRelumodel/ConC/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model/ConC/Relu?
!model/ConB3/Conv2D/ReadVariableOpReadVariableOp*model_conb3_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02#
!model/ConB3/Conv2D/ReadVariableOp?
model/ConB3/Conv2DConv2Dmodel/PoolB2/MaxPool:output:0)model/ConB3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2
model/ConB3/BiasAdd?
model/ConB3/ReluRelumodel/ConB3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(2
model/ConB3/Relu?
!model/ConA3/Conv2D/ReadVariableOpReadVariableOp*model_cona3_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02#
!model/ConA3/Conv2D/ReadVariableOp?
model/ConA3/Conv2DConv2Dmodel/PoolA2/MaxPool:output:0)model/ConA3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2
model/ConA3/BiasAdd?
model/ConA3/ReluRelumodel/ConA3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(2
model/ConA3/Relu?
model/PoolE/MaxPoolMaxPoolmodel/ConE/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
model/PoolE/MaxPool?
model/PoolD/MaxPoolMaxPoolmodel/ConD/Relu:activations:0*/
_output_shapes
:?????????		*
ksize
*
paddingVALID*
strides
2
model/PoolD/MaxPool?
model/PoolC/MaxPoolMaxPoolmodel/ConC/Relu:activations:0*/
_output_shapes
:?????????

*
ksize
*
paddingVALID*
strides
2
model/PoolC/MaxPool?
model/PoolB3/MaxPoolMaxPoolmodel/ConB3/Relu:activations:0*/
_output_shapes
:?????????(*
ksize
*
paddingVALID*
strides
2
model/PoolB3/MaxPool?
model/PoolA3/MaxPoolMaxPoolmodel/ConA3/Relu:activations:0*/
_output_shapes
:?????????(*
ksize
*
paddingVALID*
strides
2
model/PoolA3/MaxPool?
model/dropout_4/IdentityIdentitymodel/PoolE/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
model/dropout_4/Identity?
model/dropout_3/IdentityIdentitymodel/PoolD/MaxPool:output:0*
T0*/
_output_shapes
:?????????		2
model/dropout_3/Identity?
model/dropout_2/IdentityIdentitymodel/PoolC/MaxPool:output:0*
T0*/
_output_shapes
:?????????

2
model/dropout_2/Identity?
model/dropout_8/IdentityIdentitymodel/PoolB3/MaxPool:output:0*
T0*/
_output_shapes
:?????????(2
model/dropout_8/Identity?
model/dropout_7/IdentityIdentitymodel/PoolB2/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
model/dropout_7/Identity?
model/dropout_6/IdentityIdentitymodel/PoolA3/MaxPool:output:0*
T0*/
_output_shapes
:?????????(2
model/dropout_6/Identity?
model/dropout_5/IdentityIdentitymodel/PoolA2/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
model/dropout_5/Identity
model/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????`  2
model/flatten_8/Const?
model/flatten_8/ReshapeReshape!model/dropout_4/Identity:output:0model/flatten_8/Const:output:0*
T0*(
_output_shapes
:??????????2
model/flatten_8/Reshape
model/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
model/flatten_7/Const?
model/flatten_7/ReshapeReshape!model/dropout_3/Identity:output:0model/flatten_7/Const:output:0*
T0*(
_output_shapes
:??????????2
model/flatten_7/Reshape
model/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
model/flatten_6/Const?
model/flatten_6/ReshapeReshape!model/dropout_2/Identity:output:0model/flatten_6/Const:output:0*
T0*(
_output_shapes
:??????????2
model/flatten_6/Reshape
model/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????(   2
model/flatten_5/Const?
model/flatten_5/ReshapeReshape!model/dropout_8/Identity:output:0model/flatten_5/Const:output:0*
T0*'
_output_shapes
:?????????(2
model/flatten_5/Reshape
model/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
model/flatten_4/Const?
model/flatten_4/ReshapeReshape!model/dropout_7/Identity:output:0model/flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2
model/flatten_4/Reshape
model/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
model/flatten_3/Const?
model/flatten_3/ReshapeReshapemodel/PoolB/MaxPool:output:0model/flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????2
model/flatten_3/Reshape
model/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????(   2
model/flatten_2/Const?
model/flatten_2/ReshapeReshape!model/dropout_6/Identity:output:0model/flatten_2/Const:output:0*
T0*'
_output_shapes
:?????????(2
model/flatten_2/Reshape
model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
model/flatten_1/Const?
model/flatten_1/ReshapeReshape!model/dropout_5/Identity:output:0model/flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
model/flatten_1/Reshape{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????`  2
model/flatten/Const?
model/flatten/ReshapeReshapemodel/PoolA/MaxPool:output:0model/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
model/flatten/Reshape?
 model/HidA/MatMul/ReadVariableOpReadVariableOp)model_hida_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 model/HidA/MatMul/ReadVariableOp?
model/HidA/MatMulMatMulmodel/flatten/Reshape:output:0(model/HidA/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidA/MatMul?
!model/HidA/BiasAdd/ReadVariableOpReadVariableOp*model_hida_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!model/HidA/BiasAdd/ReadVariableOp?
model/HidA/BiasAddBiasAddmodel/HidA/MatMul:product:0)model/HidA/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidA/BiasAddz
model/HidA/ReluRelumodel/HidA/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidA/Relu?
!model/HidA2/MatMul/ReadVariableOpReadVariableOp*model_hida2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!model/HidA2/MatMul/ReadVariableOp?
model/HidA2/MatMulMatMul model/flatten_1/Reshape:output:0)model/HidA2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidA2/MatMul?
"model/HidA2/BiasAdd/ReadVariableOpReadVariableOp+model_hida2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"model/HidA2/BiasAdd/ReadVariableOp?
model/HidA2/BiasAddBiasAddmodel/HidA2/MatMul:product:0*model/HidA2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidA2/BiasAdd}
model/HidA2/ReluRelumodel/HidA2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidA2/Relu?
!model/HidA3/MatMul/ReadVariableOpReadVariableOp*model_hida3_matmul_readvariableop_resource*
_output_shapes
:	(?*
dtype02#
!model/HidA3/MatMul/ReadVariableOp?
model/HidA3/MatMulMatMul model/flatten_2/Reshape:output:0)model/HidA3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidA3/MatMul?
"model/HidA3/BiasAdd/ReadVariableOpReadVariableOp+model_hida3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"model/HidA3/BiasAdd/ReadVariableOp?
model/HidA3/BiasAddBiasAddmodel/HidA3/MatMul:product:0*model/HidA3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidA3/BiasAdd}
model/HidA3/ReluRelumodel/HidA3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidA3/Relu?
 model/HidB/MatMul/ReadVariableOpReadVariableOp)model_hidb_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 model/HidB/MatMul/ReadVariableOp?
model/HidB/MatMulMatMul model/flatten_3/Reshape:output:0(model/HidB/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidB/MatMul?
!model/HidB/BiasAdd/ReadVariableOpReadVariableOp*model_hidb_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!model/HidB/BiasAdd/ReadVariableOp?
model/HidB/BiasAddBiasAddmodel/HidB/MatMul:product:0)model/HidB/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidB/BiasAddz
model/HidB/ReluRelumodel/HidB/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidB/Relu?
!model/HidB2/MatMul/ReadVariableOpReadVariableOp*model_hidb2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!model/HidB2/MatMul/ReadVariableOp?
model/HidB2/MatMulMatMul model/flatten_4/Reshape:output:0)model/HidB2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidB2/MatMul?
"model/HidB2/BiasAdd/ReadVariableOpReadVariableOp+model_hidb2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"model/HidB2/BiasAdd/ReadVariableOp?
model/HidB2/BiasAddBiasAddmodel/HidB2/MatMul:product:0*model/HidB2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidB2/BiasAdd}
model/HidB2/ReluRelumodel/HidB2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidB2/Relu?
!model/HidB3/MatMul/ReadVariableOpReadVariableOp*model_hidb3_matmul_readvariableop_resource*
_output_shapes
:	(?*
dtype02#
!model/HidB3/MatMul/ReadVariableOp?
model/HidB3/MatMulMatMul model/flatten_5/Reshape:output:0)model/HidB3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidB3/MatMul?
"model/HidB3/BiasAdd/ReadVariableOpReadVariableOp+model_hidb3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"model/HidB3/BiasAdd/ReadVariableOp?
model/HidB3/BiasAddBiasAddmodel/HidB3/MatMul:product:0*model/HidB3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidB3/BiasAdd}
model/HidB3/ReluRelumodel/HidB3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidB3/Relu?
 model/HidC/MatMul/ReadVariableOpReadVariableOp)model_hidc_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 model/HidC/MatMul/ReadVariableOp?
model/HidC/MatMulMatMul model/flatten_6/Reshape:output:0(model/HidC/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidC/MatMul?
!model/HidC/BiasAdd/ReadVariableOpReadVariableOp*model_hidc_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!model/HidC/BiasAdd/ReadVariableOp?
model/HidC/BiasAddBiasAddmodel/HidC/MatMul:product:0)model/HidC/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidC/BiasAddz
model/HidC/ReluRelumodel/HidC/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidC/Relu?
!model/HidD1/MatMul/ReadVariableOpReadVariableOp*model_hidd1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!model/HidD1/MatMul/ReadVariableOp?
model/HidD1/MatMulMatMul model/flatten_7/Reshape:output:0)model/HidD1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidD1/MatMul?
"model/HidD1/BiasAdd/ReadVariableOpReadVariableOp+model_hidd1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"model/HidD1/BiasAdd/ReadVariableOp?
model/HidD1/BiasAddBiasAddmodel/HidD1/MatMul:product:0*model/HidD1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidD1/BiasAdd}
model/HidD1/ReluRelumodel/HidD1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidD1/Relu?
!model/HidE1/MatMul/ReadVariableOpReadVariableOp*model_hide1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!model/HidE1/MatMul/ReadVariableOp?
model/HidE1/MatMulMatMul model/flatten_8/Reshape:output:0)model/HidE1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidE1/MatMul?
"model/HidE1/BiasAdd/ReadVariableOpReadVariableOp+model_hide1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"model/HidE1/BiasAdd/ReadVariableOp?
model/HidE1/BiasAddBiasAddmodel/HidE1/MatMul:product:0*model/HidE1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/HidE1/BiasAdd}
model/HidE1/ReluRelumodel/HidE1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/HidE1/Relu?
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axis?
model/concatenate/concatConcatV2model/HidA/Relu:activations:0model/HidA2/Relu:activations:0model/HidA3/Relu:activations:0model/HidB/Relu:activations:0model/HidB2/Relu:activations:0model/HidB3/Relu:activations:0model/HidC/Relu:activations:0model/HidD1/Relu:activations:0model/HidE1/Relu:activations:0&model/concatenate/concat/axis:output:0*
N	*
T0*(
_output_shapes
:??????????$2
model/concatenate/concat?
 model/Hid1/MatMul/ReadVariableOpReadVariableOp)model_hid1_matmul_readvariableop_resource* 
_output_shapes
:
?$?*
dtype02"
 model/Hid1/MatMul/ReadVariableOp?
model/Hid1/MatMulMatMul!model/concatenate/concat:output:0(model/Hid1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/Hid1/MatMul?
!model/Hid1/BiasAdd/ReadVariableOpReadVariableOp*model_hid1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!model/Hid1/BiasAdd/ReadVariableOp?
model/Hid1/BiasAddBiasAddmodel/Hid1/MatMul:product:0)model/Hid1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/Hid1/BiasAddz
model/Hid1/ReluRelumodel/Hid1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/Hid1/Relu?
 model/Hid2/MatMul/ReadVariableOpReadVariableOp)model_hid2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 model/Hid2/MatMul/ReadVariableOp?
model/Hid2/MatMulMatMulmodel/Hid1/Relu:activations:0(model/Hid2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/Hid2/MatMul?
!model/Hid2/BiasAdd/ReadVariableOpReadVariableOp*model_hid2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!model/Hid2/BiasAdd/ReadVariableOp?
model/Hid2/BiasAddBiasAddmodel/Hid2/MatMul:product:0)model/Hid2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/Hid2/BiasAddz
model/Hid2/ReluRelumodel/Hid2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/Hid2/Relu?
 model/Hid3/MatMul/ReadVariableOpReadVariableOp)model_hid3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 model/Hid3/MatMul/ReadVariableOp?
model/Hid3/MatMulMatMulmodel/Hid2/Relu:activations:0(model/Hid3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/Hid3/MatMul?
!model/Hid3/BiasAdd/ReadVariableOpReadVariableOp*model_hid3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!model/Hid3/BiasAdd/ReadVariableOp?
model/Hid3/BiasAddBiasAddmodel/Hid3/MatMul:product:0)model/Hid3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/Hid3/BiasAddz
model/Hid3/TanhTanhmodel/Hid3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/Hid3/Tanh?
"model/output/MatMul/ReadVariableOpReadVariableOp+model_output_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02$
"model/output/MatMul/ReadVariableOp?
model/output/MatMulMatMulmodel/Hid3/Tanh:y:0*model/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
model/output/MatMul?
#model/output/BiasAdd/ReadVariableOpReadVariableOp,model_output_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02%
#model/output/BiasAdd/ReadVariableOp?
model/output/BiasAddBiasAddmodel/output/MatMul:product:0+model/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
model/output/BiasAdd?
model/output/SoftmaxSoftmaxmodel/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
model/output/Softmaxy
IdentityIdentitymodel/output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp"^model/ConA/BiasAdd/ReadVariableOp!^model/ConA/Conv2D/ReadVariableOp#^model/ConA2/BiasAdd/ReadVariableOp"^model/ConA2/Conv2D/ReadVariableOp#^model/ConA3/BiasAdd/ReadVariableOp"^model/ConA3/Conv2D/ReadVariableOp"^model/ConB/BiasAdd/ReadVariableOp!^model/ConB/Conv2D/ReadVariableOp#^model/ConB2/BiasAdd/ReadVariableOp"^model/ConB2/Conv2D/ReadVariableOp#^model/ConB3/BiasAdd/ReadVariableOp"^model/ConB3/Conv2D/ReadVariableOp"^model/ConC/BiasAdd/ReadVariableOp!^model/ConC/Conv2D/ReadVariableOp"^model/ConD/BiasAdd/ReadVariableOp!^model/ConD/Conv2D/ReadVariableOp"^model/ConE/BiasAdd/ReadVariableOp!^model/ConE/Conv2D/ReadVariableOp"^model/Hid1/BiasAdd/ReadVariableOp!^model/Hid1/MatMul/ReadVariableOp"^model/Hid2/BiasAdd/ReadVariableOp!^model/Hid2/MatMul/ReadVariableOp"^model/Hid3/BiasAdd/ReadVariableOp!^model/Hid3/MatMul/ReadVariableOp"^model/HidA/BiasAdd/ReadVariableOp!^model/HidA/MatMul/ReadVariableOp#^model/HidA2/BiasAdd/ReadVariableOp"^model/HidA2/MatMul/ReadVariableOp#^model/HidA3/BiasAdd/ReadVariableOp"^model/HidA3/MatMul/ReadVariableOp"^model/HidB/BiasAdd/ReadVariableOp!^model/HidB/MatMul/ReadVariableOp#^model/HidB2/BiasAdd/ReadVariableOp"^model/HidB2/MatMul/ReadVariableOp#^model/HidB3/BiasAdd/ReadVariableOp"^model/HidB3/MatMul/ReadVariableOp"^model/HidC/BiasAdd/ReadVariableOp!^model/HidC/MatMul/ReadVariableOp#^model/HidD1/BiasAdd/ReadVariableOp"^model/HidD1/MatMul/ReadVariableOp#^model/HidE1/BiasAdd/ReadVariableOp"^model/HidE1/MatMul/ReadVariableOp$^model/output/BiasAdd/ReadVariableOp#^model/output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
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
 model/ConC/Conv2D/ReadVariableOp model/ConC/Conv2D/ReadVariableOp2F
!model/ConD/BiasAdd/ReadVariableOp!model/ConD/BiasAdd/ReadVariableOp2D
 model/ConD/Conv2D/ReadVariableOp model/ConD/Conv2D/ReadVariableOp2F
!model/ConE/BiasAdd/ReadVariableOp!model/ConE/BiasAdd/ReadVariableOp2D
 model/ConE/Conv2D/ReadVariableOp model/ConE/Conv2D/ReadVariableOp2F
!model/Hid1/BiasAdd/ReadVariableOp!model/Hid1/BiasAdd/ReadVariableOp2D
 model/Hid1/MatMul/ReadVariableOp model/Hid1/MatMul/ReadVariableOp2F
!model/Hid2/BiasAdd/ReadVariableOp!model/Hid2/BiasAdd/ReadVariableOp2D
 model/Hid2/MatMul/ReadVariableOp model/Hid2/MatMul/ReadVariableOp2F
!model/Hid3/BiasAdd/ReadVariableOp!model/Hid3/BiasAdd/ReadVariableOp2D
 model/Hid3/MatMul/ReadVariableOp model/Hid3/MatMul/ReadVariableOp2F
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
"model/HidD1/BiasAdd/ReadVariableOp"model/HidD1/BiasAdd/ReadVariableOp2F
!model/HidD1/MatMul/ReadVariableOp!model/HidD1/MatMul/ReadVariableOp2H
"model/HidE1/BiasAdd/ReadVariableOp"model/HidE1/BiasAdd/ReadVariableOp2F
!model/HidE1/MatMul/ReadVariableOp!model/HidE1/MatMul/ReadVariableOp2J
#model/output/BiasAdd/ReadVariableOp#model/output/BiasAdd/ReadVariableOp2H
"model/output/MatMul/ReadVariableOp"model/output/MatMul/ReadVariableOp:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?
`
'__inference_dropout_layer_call_fn_23276

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_216832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
\
@__inference_PoolD_layer_call_and_return_conditional_losses_20486

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
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
c
D__inference_dropout_3_layer_call_and_return_conditional_losses_23762

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
:?????????		2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????		*
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
:?????????		2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????		2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????		2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????		2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????		:W S
/
_output_shapes
:?????????		
 
_user_specified_nameinputs
?
?
@__inference_HidA2_layer_call_and_return_conditional_losses_20898

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
@__inference_HidA3_layer_call_and_return_conditional_losses_23948

inputs1
matmul_readvariableop_resource:	(?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(?*
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
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
\
@__inference_PoolE_layer_call_and_return_conditional_losses_20723

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_4_layer_call_fn_23772

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_215552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
]
A__inference_PoolA3_layer_call_and_return_conditional_losses_20747

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????(*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_22545	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:		

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:(

unknown_14:($

unknown_15:(

unknown_16:(

unknown_17:
??

unknown_18:	?

unknown_19:
??

unknown_20:	?

unknown_21:	(?

unknown_22:	?

unknown_23:
??

unknown_24:	?

unknown_25:
??

unknown_26:	?

unknown_27:	(?

unknown_28:	?

unknown_29:
??

unknown_30:	?

unknown_31:
??

unknown_32:	?

unknown_33:
??

unknown_34:	?

unknown_35:
?$?

unknown_36:	?

unknown_37:
??

unknown_38:	?

unknown_39:
??

unknown_40:	?

unknown_41:	?


unknown_42:

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
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_203232
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?
c
D__inference_dropout_8_layer_call_and_return_conditional_losses_21486

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
:?????????(2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????(2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????(2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
\
@__inference_PoolE_layer_call_and_return_conditional_losses_23600

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_5_layer_call_fn_23610

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_214172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_4_layer_call_and_return_conditional_losses_23844

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_ConB2_layer_call_fn_23349

inputs!
unknown:
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
:?????????		*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConB2_layer_call_and_return_conditional_losses_205992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????		2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_20754

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
?__inference_ConE_layer_call_and_return_conditional_losses_20645

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
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
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_concatenate_layer_call_and_return_conditional_losses_21037

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8concat/axis:output:0*
N	*
T0*(
_output_shapes
:??????????$2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_6_layer_call_fn_23632

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
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_207892
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
@__inference_HidB3_layer_call_and_return_conditional_losses_20966

inputs1
matmul_readvariableop_resource:	(?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(?*
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
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
@__inference_HidE1_layer_call_and_return_conditional_losses_24068

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
\
@__inference_PoolE_layer_call_and_return_conditional_losses_23595

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
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
%__inference_ConA2_layer_call_fn_23329

inputs!
unknown:
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
:?????????

*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConA2_layer_call_and_return_conditional_losses_206162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????

2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_20844

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
@__inference_model_layer_call_and_return_conditional_losses_21976

inputs$

conb_21837:

conb_21839:$

cona_21842:

cona_21844:%
conb2_21851:
conb2_21853:%
cona2_21856:
cona2_21858:$

cone_21863:

cone_21865:$

cond_21868:		

cond_21870:$

conc_21873:

conc_21875:%
conb3_21878:(
conb3_21880:(%
cona3_21883:(
cona3_21885:(

hida_21909:
??

hida_21911:	?
hida2_21914:
??
hida2_21916:	?
hida3_21919:	(?
hida3_21921:	?

hidb_21924:
??

hidb_21926:	?
hidb2_21929:
??
hidb2_21931:	?
hidb3_21934:	(?
hidb3_21936:	?

hidc_21939:
??

hidc_21941:	?
hidd1_21944:
??
hidd1_21946:	?
hide1_21949:
??
hide1_21951:	?

hid1_21955:
?$?

hid1_21957:	?

hid2_21960:
??

hid2_21962:	?

hid3_21965:
??

hid3_21967:	?
output_21970:	?

output_21972:

identity??ConA/StatefulPartitionedCall?ConA2/StatefulPartitionedCall?ConA3/StatefulPartitionedCall?ConB/StatefulPartitionedCall?ConB2/StatefulPartitionedCall?ConB3/StatefulPartitionedCall?ConC/StatefulPartitionedCall?ConD/StatefulPartitionedCall?ConE/StatefulPartitionedCall?Hid1/StatefulPartitionedCall?Hid2/StatefulPartitionedCall?Hid3/StatefulPartitionedCall?HidA/StatefulPartitionedCall?HidA2/StatefulPartitionedCall?HidA3/StatefulPartitionedCall?HidB/StatefulPartitionedCall?HidB2/StatefulPartitionedCall?HidB3/StatefulPartitionedCall?HidC/StatefulPartitionedCall?HidD1/StatefulPartitionedCall?HidE1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?!dropout_5/StatefulPartitionedCall?!dropout_6/StatefulPartitionedCall?!dropout_7/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?output/StatefulPartitionedCall?
ConB/StatefulPartitionedCallStatefulPartitionedCallinputs
conb_21837
conb_21839*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_ConB_layer_call_and_return_conditional_losses_205392
ConB/StatefulPartitionedCall?
ConA/StatefulPartitionedCallStatefulPartitionedCallinputs
cona_21842
cona_21844*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_ConA_layer_call_and_return_conditional_losses_205562
ConA/StatefulPartitionedCall?
PoolB/PartitionedCallPartitionedCall%ConB/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_PoolB_layer_call_and_return_conditional_losses_205662
PoolB/PartitionedCall?
PoolA/PartitionedCallPartitionedCall%ConA/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_PoolA_layer_call_and_return_conditional_losses_205722
PoolA/PartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCallPoolB/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_217062#
!dropout_1/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCallPoolA/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_216832!
dropout/StatefulPartitionedCall?
ConB2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conb2_21851conb2_21853*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConB2_layer_call_and_return_conditional_losses_205992
ConB2/StatefulPartitionedCall?
ConA2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0cona2_21856cona2_21858*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConA2_layer_call_and_return_conditional_losses_206162
ConA2/StatefulPartitionedCall?
PoolB2/PartitionedCallPartitionedCall&ConB2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolB2_layer_call_and_return_conditional_losses_206262
PoolB2/PartitionedCall?
PoolA2/PartitionedCallPartitionedCall&ConA2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolA2_layer_call_and_return_conditional_losses_206322
PoolA2/PartitionedCall?
ConE/StatefulPartitionedCallStatefulPartitionedCallinputs
cone_21863
cone_21865*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_ConE_layer_call_and_return_conditional_losses_206452
ConE/StatefulPartitionedCall?
ConD/StatefulPartitionedCallStatefulPartitionedCallinputs
cond_21868
cond_21870*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_ConD_layer_call_and_return_conditional_losses_206622
ConD/StatefulPartitionedCall?
ConC/StatefulPartitionedCallStatefulPartitionedCallinputs
conc_21873
conc_21875*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_ConC_layer_call_and_return_conditional_losses_206792
ConC/StatefulPartitionedCall?
ConB3/StatefulPartitionedCallStatefulPartitionedCallPoolB2/PartitionedCall:output:0conb3_21878conb3_21880*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConB3_layer_call_and_return_conditional_losses_206962
ConB3/StatefulPartitionedCall?
ConA3/StatefulPartitionedCallStatefulPartitionedCallPoolA2/PartitionedCall:output:0cona3_21883cona3_21885*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConA3_layer_call_and_return_conditional_losses_207132
ConA3/StatefulPartitionedCall?
PoolE/PartitionedCallPartitionedCall%ConE/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_PoolE_layer_call_and_return_conditional_losses_207232
PoolE/PartitionedCall?
PoolD/PartitionedCallPartitionedCall%ConD/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_PoolD_layer_call_and_return_conditional_losses_207292
PoolD/PartitionedCall?
PoolC/PartitionedCallPartitionedCall%ConC/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_PoolC_layer_call_and_return_conditional_losses_207352
PoolC/PartitionedCall?
PoolB3/PartitionedCallPartitionedCall&ConB3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolB3_layer_call_and_return_conditional_losses_207412
PoolB3/PartitionedCall?
PoolA3/PartitionedCallPartitionedCall&ConA3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolA3_layer_call_and_return_conditional_losses_207472
PoolA3/PartitionedCall?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCallPoolE/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_215552#
!dropout_4/StatefulPartitionedCall?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCallPoolD/PartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_215322#
!dropout_3/StatefulPartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCallPoolC/PartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_215092#
!dropout_2/StatefulPartitionedCall?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCallPoolB3/PartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_214862#
!dropout_8/StatefulPartitionedCall?
!dropout_7/StatefulPartitionedCallStatefulPartitionedCallPoolB2/PartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_214632#
!dropout_7/StatefulPartitionedCall?
!dropout_6/StatefulPartitionedCallStatefulPartitionedCallPoolA3/PartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_214402#
!dropout_6/StatefulPartitionedCall?
!dropout_5/StatefulPartitionedCallStatefulPartitionedCallPoolA2/PartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_214172#
!dropout_5/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCall*dropout_4/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *M
fHRF
D__inference_flatten_8_layer_call_and_return_conditional_losses_208042
flatten_8/PartitionedCall?
flatten_7/PartitionedCallPartitionedCall*dropout_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_7_layer_call_and_return_conditional_losses_208122
flatten_7/PartitionedCall?
flatten_6/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_208202
flatten_6/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall*dropout_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_208282
flatten_5/PartitionedCall?
flatten_4/PartitionedCallPartitionedCall*dropout_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_208362
flatten_4/PartitionedCall?
flatten_3/PartitionedCallPartitionedCallPoolB/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_208442
flatten_3/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall*dropout_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_208522
flatten_2/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall*dropout_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_208602
flatten_1/PartitionedCall?
flatten/PartitionedCallPartitionedCallPoolA/PartitionedCall:output:0*
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
GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_208682
flatten/PartitionedCall?
HidA/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
hida_21909
hida_21911*
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
GPU2*0J 8? *H
fCRA
?__inference_HidA_layer_call_and_return_conditional_losses_208812
HidA/StatefulPartitionedCall?
HidA2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0hida2_21914hida2_21916*
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
@__inference_HidA2_layer_call_and_return_conditional_losses_208982
HidA2/StatefulPartitionedCall?
HidA3/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0hida3_21919hida3_21921*
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
@__inference_HidA3_layer_call_and_return_conditional_losses_209152
HidA3/StatefulPartitionedCall?
HidB/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0
hidb_21924
hidb_21926*
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
GPU2*0J 8? *H
fCRA
?__inference_HidB_layer_call_and_return_conditional_losses_209322
HidB/StatefulPartitionedCall?
HidB2/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0hidb2_21929hidb2_21931*
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
@__inference_HidB2_layer_call_and_return_conditional_losses_209492
HidB2/StatefulPartitionedCall?
HidB3/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0hidb3_21934hidb3_21936*
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
@__inference_HidB3_layer_call_and_return_conditional_losses_209662
HidB3/StatefulPartitionedCall?
HidC/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0
hidc_21939
hidc_21941*
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
GPU2*0J 8? *H
fCRA
?__inference_HidC_layer_call_and_return_conditional_losses_209832
HidC/StatefulPartitionedCall?
HidD1/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0hidd1_21944hidd1_21946*
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
@__inference_HidD1_layer_call_and_return_conditional_losses_210002
HidD1/StatefulPartitionedCall?
HidE1/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0hide1_21949hide1_21951*
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
@__inference_HidE1_layer_call_and_return_conditional_losses_210172
HidE1/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall%HidA/StatefulPartitionedCall:output:0&HidA2/StatefulPartitionedCall:output:0&HidA3/StatefulPartitionedCall:output:0%HidB/StatefulPartitionedCall:output:0&HidB2/StatefulPartitionedCall:output:0&HidB3/StatefulPartitionedCall:output:0%HidC/StatefulPartitionedCall:output:0&HidD1/StatefulPartitionedCall:output:0&HidE1/StatefulPartitionedCall:output:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_210372
concatenate/PartitionedCall?
Hid1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0
hid1_21955
hid1_21957*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_Hid1_layer_call_and_return_conditional_losses_210502
Hid1/StatefulPartitionedCall?
Hid2/StatefulPartitionedCallStatefulPartitionedCall%Hid1/StatefulPartitionedCall:output:0
hid2_21960
hid2_21962*
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
GPU2*0J 8? *H
fCRA
?__inference_Hid2_layer_call_and_return_conditional_losses_210672
Hid2/StatefulPartitionedCall?
Hid3/StatefulPartitionedCallStatefulPartitionedCall%Hid2/StatefulPartitionedCall:output:0
hid3_21965
hid3_21967*
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
GPU2*0J 8? *H
fCRA
?__inference_Hid3_layer_call_and_return_conditional_losses_210842
Hid3/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall%Hid3/StatefulPartitionedCall:output:0output_21970output_21972*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_211012 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp^ConA/StatefulPartitionedCall^ConA2/StatefulPartitionedCall^ConA3/StatefulPartitionedCall^ConB/StatefulPartitionedCall^ConB2/StatefulPartitionedCall^ConB3/StatefulPartitionedCall^ConC/StatefulPartitionedCall^ConD/StatefulPartitionedCall^ConE/StatefulPartitionedCall^Hid1/StatefulPartitionedCall^Hid2/StatefulPartitionedCall^Hid3/StatefulPartitionedCall^HidA/StatefulPartitionedCall^HidA2/StatefulPartitionedCall^HidA3/StatefulPartitionedCall^HidB/StatefulPartitionedCall^HidB2/StatefulPartitionedCall^HidB3/StatefulPartitionedCall^HidC/StatefulPartitionedCall^HidD1/StatefulPartitionedCall^HidE1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
ConA/StatefulPartitionedCallConA/StatefulPartitionedCall2>
ConA2/StatefulPartitionedCallConA2/StatefulPartitionedCall2>
ConA3/StatefulPartitionedCallConA3/StatefulPartitionedCall2<
ConB/StatefulPartitionedCallConB/StatefulPartitionedCall2>
ConB2/StatefulPartitionedCallConB2/StatefulPartitionedCall2>
ConB3/StatefulPartitionedCallConB3/StatefulPartitionedCall2<
ConC/StatefulPartitionedCallConC/StatefulPartitionedCall2<
ConD/StatefulPartitionedCallConD/StatefulPartitionedCall2<
ConE/StatefulPartitionedCallConE/StatefulPartitionedCall2<
Hid1/StatefulPartitionedCallHid1/StatefulPartitionedCall2<
Hid2/StatefulPartitionedCallHid2/StatefulPartitionedCall2<
Hid3/StatefulPartitionedCallHid3/StatefulPartitionedCall2<
HidA/StatefulPartitionedCallHidA/StatefulPartitionedCall2>
HidA2/StatefulPartitionedCallHidA2/StatefulPartitionedCall2>
HidA3/StatefulPartitionedCallHidA3/StatefulPartitionedCall2<
HidB/StatefulPartitionedCallHidB/StatefulPartitionedCall2>
HidB2/StatefulPartitionedCallHidB2/StatefulPartitionedCall2>
HidB3/StatefulPartitionedCallHidB3/StatefulPartitionedCall2<
HidC/StatefulPartitionedCallHidC/StatefulPartitionedCall2>
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
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
@__inference_ConA2_layer_call_and_return_conditional_losses_23340

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
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
:?????????

2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????

2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????

2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_3_layer_call_fn_23827

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
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_208442
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
\
@__inference_PoolE_layer_call_and_return_conditional_losses_20508

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
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
?__inference_Hid1_layer_call_and_return_conditional_losses_21050

inputs2
matmul_readvariableop_resource:
?$?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?$?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
?
E
)__inference_flatten_7_layer_call_fn_23871

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
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_7_layer_call_and_return_conditional_losses_208122
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????		:W S
/
_output_shapes
:?????????		
 
_user_specified_nameinputs
?
]
A__inference_PoolA3_layer_call_and_return_conditional_losses_23515

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
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_23735

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
:?????????

2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????

*
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
:?????????

2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????

2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????

2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????

2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_23281

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_2_layer_call_fn_23718

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
:?????????

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_215092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????

2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_23777

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
@__inference_HidA3_layer_call_and_return_conditional_losses_20915

inputs1
matmul_readvariableop_resource:	(?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(?*
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
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
B
&__inference_PoolA3_layer_call_fn_23510

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
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolA3_layer_call_and_return_conditional_losses_207472
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
\
@__inference_PoolB_layer_call_and_return_conditional_losses_23266

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_23789

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
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
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
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_22731

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:		

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:(

unknown_14:($

unknown_15:(

unknown_16:(

unknown_17:
??

unknown_18:	?

unknown_19:
??

unknown_20:	?

unknown_21:	(?

unknown_22:	?

unknown_23:
??

unknown_24:	?

unknown_25:
??

unknown_26:	?

unknown_27:	(?

unknown_28:	?

unknown_29:
??

unknown_30:	?

unknown_31:
??

unknown_32:	?

unknown_33:
??

unknown_34:	?

unknown_35:
?$?

unknown_36:	?

unknown_37:
??

unknown_38:	?

unknown_39:
??

unknown_40:	?

unknown_41:	?


unknown_42:

identity??StatefulPartitionedCall?
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
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_219762
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_23822

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????(   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????(2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs

?5
__inference__traced_save_24621
file_prefix*
&savev2_cona_kernel_read_readvariableop(
$savev2_cona_bias_read_readvariableop*
&savev2_conb_kernel_read_readvariableop(
$savev2_conb_bias_read_readvariableop+
'savev2_cona2_kernel_read_readvariableop)
%savev2_cona2_bias_read_readvariableop+
'savev2_conb2_kernel_read_readvariableop)
%savev2_conb2_bias_read_readvariableop+
'savev2_cona3_kernel_read_readvariableop)
%savev2_cona3_bias_read_readvariableop+
'savev2_conb3_kernel_read_readvariableop)
%savev2_conb3_bias_read_readvariableop*
&savev2_conc_kernel_read_readvariableop(
$savev2_conc_bias_read_readvariableop*
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
'savev2_hidd1_kernel_read_readvariableop)
%savev2_hidd1_bias_read_readvariableop+
'savev2_hide1_kernel_read_readvariableop)
%savev2_hide1_bias_read_readvariableop*
&savev2_hid1_kernel_read_readvariableop(
$savev2_hid1_bias_read_readvariableop*
&savev2_hid2_kernel_read_readvariableop(
$savev2_hid2_bias_read_readvariableop*
&savev2_hid3_kernel_read_readvariableop(
$savev2_hid3_bias_read_readvariableop,
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
,savev2_adam_cona2_bias_m_read_readvariableop2
.savev2_adam_conb2_kernel_m_read_readvariableop0
,savev2_adam_conb2_bias_m_read_readvariableop2
.savev2_adam_cona3_kernel_m_read_readvariableop0
,savev2_adam_cona3_bias_m_read_readvariableop2
.savev2_adam_conb3_kernel_m_read_readvariableop0
,savev2_adam_conb3_bias_m_read_readvariableop1
-savev2_adam_conc_kernel_m_read_readvariableop/
+savev2_adam_conc_bias_m_read_readvariableop1
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
.savev2_adam_hidd1_kernel_m_read_readvariableop0
,savev2_adam_hidd1_bias_m_read_readvariableop2
.savev2_adam_hide1_kernel_m_read_readvariableop0
,savev2_adam_hide1_bias_m_read_readvariableop1
-savev2_adam_hid1_kernel_m_read_readvariableop/
+savev2_adam_hid1_bias_m_read_readvariableop1
-savev2_adam_hid2_kernel_m_read_readvariableop/
+savev2_adam_hid2_bias_m_read_readvariableop1
-savev2_adam_hid3_kernel_m_read_readvariableop/
+savev2_adam_hid3_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop1
-savev2_adam_cona_kernel_v_read_readvariableop/
+savev2_adam_cona_bias_v_read_readvariableop1
-savev2_adam_conb_kernel_v_read_readvariableop/
+savev2_adam_conb_bias_v_read_readvariableop2
.savev2_adam_cona2_kernel_v_read_readvariableop0
,savev2_adam_cona2_bias_v_read_readvariableop2
.savev2_adam_conb2_kernel_v_read_readvariableop0
,savev2_adam_conb2_bias_v_read_readvariableop2
.savev2_adam_cona3_kernel_v_read_readvariableop0
,savev2_adam_cona3_bias_v_read_readvariableop2
.savev2_adam_conb3_kernel_v_read_readvariableop0
,savev2_adam_conb3_bias_v_read_readvariableop1
-savev2_adam_conc_kernel_v_read_readvariableop/
+savev2_adam_conc_bias_v_read_readvariableop1
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
.savev2_adam_hidd1_kernel_v_read_readvariableop0
,savev2_adam_hidd1_bias_v_read_readvariableop2
.savev2_adam_hide1_kernel_v_read_readvariableop0
,savev2_adam_hide1_bias_v_read_readvariableop1
-savev2_adam_hid1_kernel_v_read_readvariableop/
+savev2_adam_hid1_bias_v_read_readvariableop1
-savev2_adam_hid2_kernel_v_read_readvariableop/
+savev2_adam_hid2_bias_v_read_readvariableop1
-savev2_adam_hid3_kernel_v_read_readvariableop/
+savev2_adam_hid3_bias_v_read_readvariableop3
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
ShardedFilename?Q
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?P
value?PB?P?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?2
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_cona_kernel_read_readvariableop$savev2_cona_bias_read_readvariableop&savev2_conb_kernel_read_readvariableop$savev2_conb_bias_read_readvariableop'savev2_cona2_kernel_read_readvariableop%savev2_cona2_bias_read_readvariableop'savev2_conb2_kernel_read_readvariableop%savev2_conb2_bias_read_readvariableop'savev2_cona3_kernel_read_readvariableop%savev2_cona3_bias_read_readvariableop'savev2_conb3_kernel_read_readvariableop%savev2_conb3_bias_read_readvariableop&savev2_conc_kernel_read_readvariableop$savev2_conc_bias_read_readvariableop&savev2_cond_kernel_read_readvariableop$savev2_cond_bias_read_readvariableop&savev2_cone_kernel_read_readvariableop$savev2_cone_bias_read_readvariableop&savev2_hida_kernel_read_readvariableop$savev2_hida_bias_read_readvariableop'savev2_hida2_kernel_read_readvariableop%savev2_hida2_bias_read_readvariableop'savev2_hida3_kernel_read_readvariableop%savev2_hida3_bias_read_readvariableop&savev2_hidb_kernel_read_readvariableop$savev2_hidb_bias_read_readvariableop'savev2_hidb2_kernel_read_readvariableop%savev2_hidb2_bias_read_readvariableop'savev2_hidb3_kernel_read_readvariableop%savev2_hidb3_bias_read_readvariableop&savev2_hidc_kernel_read_readvariableop$savev2_hidc_bias_read_readvariableop'savev2_hidd1_kernel_read_readvariableop%savev2_hidd1_bias_read_readvariableop'savev2_hide1_kernel_read_readvariableop%savev2_hide1_bias_read_readvariableop&savev2_hid1_kernel_read_readvariableop$savev2_hid1_bias_read_readvariableop&savev2_hid2_kernel_read_readvariableop$savev2_hid2_bias_read_readvariableop&savev2_hid3_kernel_read_readvariableop$savev2_hid3_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop-savev2_adam_cona_kernel_m_read_readvariableop+savev2_adam_cona_bias_m_read_readvariableop-savev2_adam_conb_kernel_m_read_readvariableop+savev2_adam_conb_bias_m_read_readvariableop.savev2_adam_cona2_kernel_m_read_readvariableop,savev2_adam_cona2_bias_m_read_readvariableop.savev2_adam_conb2_kernel_m_read_readvariableop,savev2_adam_conb2_bias_m_read_readvariableop.savev2_adam_cona3_kernel_m_read_readvariableop,savev2_adam_cona3_bias_m_read_readvariableop.savev2_adam_conb3_kernel_m_read_readvariableop,savev2_adam_conb3_bias_m_read_readvariableop-savev2_adam_conc_kernel_m_read_readvariableop+savev2_adam_conc_bias_m_read_readvariableop-savev2_adam_cond_kernel_m_read_readvariableop+savev2_adam_cond_bias_m_read_readvariableop-savev2_adam_cone_kernel_m_read_readvariableop+savev2_adam_cone_bias_m_read_readvariableop-savev2_adam_hida_kernel_m_read_readvariableop+savev2_adam_hida_bias_m_read_readvariableop.savev2_adam_hida2_kernel_m_read_readvariableop,savev2_adam_hida2_bias_m_read_readvariableop.savev2_adam_hida3_kernel_m_read_readvariableop,savev2_adam_hida3_bias_m_read_readvariableop-savev2_adam_hidb_kernel_m_read_readvariableop+savev2_adam_hidb_bias_m_read_readvariableop.savev2_adam_hidb2_kernel_m_read_readvariableop,savev2_adam_hidb2_bias_m_read_readvariableop.savev2_adam_hidb3_kernel_m_read_readvariableop,savev2_adam_hidb3_bias_m_read_readvariableop-savev2_adam_hidc_kernel_m_read_readvariableop+savev2_adam_hidc_bias_m_read_readvariableop.savev2_adam_hidd1_kernel_m_read_readvariableop,savev2_adam_hidd1_bias_m_read_readvariableop.savev2_adam_hide1_kernel_m_read_readvariableop,savev2_adam_hide1_bias_m_read_readvariableop-savev2_adam_hid1_kernel_m_read_readvariableop+savev2_adam_hid1_bias_m_read_readvariableop-savev2_adam_hid2_kernel_m_read_readvariableop+savev2_adam_hid2_bias_m_read_readvariableop-savev2_adam_hid3_kernel_m_read_readvariableop+savev2_adam_hid3_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop-savev2_adam_cona_kernel_v_read_readvariableop+savev2_adam_cona_bias_v_read_readvariableop-savev2_adam_conb_kernel_v_read_readvariableop+savev2_adam_conb_bias_v_read_readvariableop.savev2_adam_cona2_kernel_v_read_readvariableop,savev2_adam_cona2_bias_v_read_readvariableop.savev2_adam_conb2_kernel_v_read_readvariableop,savev2_adam_conb2_bias_v_read_readvariableop.savev2_adam_cona3_kernel_v_read_readvariableop,savev2_adam_cona3_bias_v_read_readvariableop.savev2_adam_conb3_kernel_v_read_readvariableop,savev2_adam_conb3_bias_v_read_readvariableop-savev2_adam_conc_kernel_v_read_readvariableop+savev2_adam_conc_bias_v_read_readvariableop-savev2_adam_cond_kernel_v_read_readvariableop+savev2_adam_cond_bias_v_read_readvariableop-savev2_adam_cone_kernel_v_read_readvariableop+savev2_adam_cone_bias_v_read_readvariableop-savev2_adam_hida_kernel_v_read_readvariableop+savev2_adam_hida_bias_v_read_readvariableop.savev2_adam_hida2_kernel_v_read_readvariableop,savev2_adam_hida2_bias_v_read_readvariableop.savev2_adam_hida3_kernel_v_read_readvariableop,savev2_adam_hida3_bias_v_read_readvariableop-savev2_adam_hidb_kernel_v_read_readvariableop+savev2_adam_hidb_bias_v_read_readvariableop.savev2_adam_hidb2_kernel_v_read_readvariableop,savev2_adam_hidb2_bias_v_read_readvariableop.savev2_adam_hidb3_kernel_v_read_readvariableop,savev2_adam_hidb3_bias_v_read_readvariableop-savev2_adam_hidc_kernel_v_read_readvariableop+savev2_adam_hidc_bias_v_read_readvariableop.savev2_adam_hidd1_kernel_v_read_readvariableop,savev2_adam_hidd1_bias_v_read_readvariableop.savev2_adam_hide1_kernel_v_read_readvariableop,savev2_adam_hide1_bias_v_read_readvariableop-savev2_adam_hid1_kernel_v_read_readvariableop+savev2_adam_hid1_bias_v_read_readvariableop-savev2_adam_hid2_kernel_v_read_readvariableop+savev2_adam_hid2_bias_v_read_readvariableop-savev2_adam_hid3_kernel_v_read_readvariableop+savev2_adam_hid3_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes?
?2?	2
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

?
: :::::::::(:(:(:(:::		::::
??:?:
??:?:	(?:?:
??:?:
??:?:	(?:?:
??:?:
??:?:
??:?:
?$?:?:
??:?:
??:?:	?
:
: : : : : : : : : :::::::::(:(:(:(:::		::::
??:?:
??:?:	(?:?:
??:?:
??:?:	(?:?:
??:?:
??:?:
??:?:
?$?:?:
??:?:
??:?:	?
:
:::::::::(:(:(:(:::		::::
??:?:
??:?:	(?:?:
??:?:
??:?:	(?:?:
??:?:
??:?:
??:?:
?$?:?:
??:?:
??:?:	?
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 
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
:(: 

_output_shapes
:(:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:		: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	(?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	(?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:! 

_output_shapes	
:?:&!"
 
_output_shapes
:
??:!"

_output_shapes	
:?:&#"
 
_output_shapes
:
??:!$

_output_shapes	
:?:&%"
 
_output_shapes
:
?$?:!&

_output_shapes	
:?:&'"
 
_output_shapes
:
??:!(

_output_shapes	
:?:&)"
 
_output_shapes
:
??:!*

_output_shapes	
:?:%+!

_output_shapes
:	?
: ,

_output_shapes
:
:-

_output_shapes
: :.

_output_shapes
: :/
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
: :,6(
&
_output_shapes
:: 7

_output_shapes
::,8(
&
_output_shapes
:: 9

_output_shapes
::,:(
&
_output_shapes
:: ;

_output_shapes
::,<(
&
_output_shapes
:: =

_output_shapes
::,>(
&
_output_shapes
:(: ?

_output_shapes
:(:,@(
&
_output_shapes
:(: A

_output_shapes
:(:,B(
&
_output_shapes
:: C

_output_shapes
::,D(
&
_output_shapes
:		: E

_output_shapes
::,F(
&
_output_shapes
:: G

_output_shapes
::&H"
 
_output_shapes
:
??:!I

_output_shapes	
:?:&J"
 
_output_shapes
:
??:!K

_output_shapes	
:?:%L!

_output_shapes
:	(?:!M

_output_shapes	
:?:&N"
 
_output_shapes
:
??:!O

_output_shapes	
:?:&P"
 
_output_shapes
:
??:!Q

_output_shapes	
:?:%R!

_output_shapes
:	(?:!S

_output_shapes	
:?:&T"
 
_output_shapes
:
??:!U

_output_shapes	
:?:&V"
 
_output_shapes
:
??:!W

_output_shapes	
:?:&X"
 
_output_shapes
:
??:!Y

_output_shapes	
:?:&Z"
 
_output_shapes
:
?$?:![

_output_shapes	
:?:&\"
 
_output_shapes
:
??:!]

_output_shapes	
:?:&^"
 
_output_shapes
:
??:!_

_output_shapes	
:?:%`!

_output_shapes
:	?
: a

_output_shapes
:
:,b(
&
_output_shapes
:: c

_output_shapes
::,d(
&
_output_shapes
:: e

_output_shapes
::,f(
&
_output_shapes
:: g

_output_shapes
::,h(
&
_output_shapes
:: i

_output_shapes
::,j(
&
_output_shapes
:(: k

_output_shapes
:(:,l(
&
_output_shapes
:(: m

_output_shapes
:(:,n(
&
_output_shapes
:: o

_output_shapes
::,p(
&
_output_shapes
:		: q

_output_shapes
::,r(
&
_output_shapes
:: s

_output_shapes
::&t"
 
_output_shapes
:
??:!u

_output_shapes	
:?:&v"
 
_output_shapes
:
??:!w

_output_shapes	
:?:%x!

_output_shapes
:	(?:!y

_output_shapes	
:?:&z"
 
_output_shapes
:
??:!{

_output_shapes	
:?:&|"
 
_output_shapes
:
??:!}

_output_shapes	
:?:%~!

_output_shapes
:	(?:!

_output_shapes	
:?:'?"
 
_output_shapes
:
??:"?

_output_shapes	
:?:'?"
 
_output_shapes
:
??:"?

_output_shapes	
:?:'?"
 
_output_shapes
:
??:"?

_output_shapes	
:?:'?"
 
_output_shapes
:
?$?:"?

_output_shapes	
:?:'?"
 
_output_shapes
:
??:"?
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
:	?
:!?

_output_shapes
:
:?

_output_shapes
: 
?
?
%__inference_HidA3_layer_call_fn_23937

inputs
unknown:	(?
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
@__inference_HidA3_layer_call_and_return_conditional_losses_209152
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
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
E
)__inference_dropout_4_layer_call_fn_23767

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_207542
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
@__inference_model_layer_call_and_return_conditional_losses_22927

inputs=
#conb_conv2d_readvariableop_resource:2
$conb_biasadd_readvariableop_resource:=
#cona_conv2d_readvariableop_resource:2
$cona_biasadd_readvariableop_resource:>
$conb2_conv2d_readvariableop_resource:3
%conb2_biasadd_readvariableop_resource:>
$cona2_conv2d_readvariableop_resource:3
%cona2_biasadd_readvariableop_resource:=
#cone_conv2d_readvariableop_resource:2
$cone_biasadd_readvariableop_resource:=
#cond_conv2d_readvariableop_resource:		2
$cond_biasadd_readvariableop_resource:=
#conc_conv2d_readvariableop_resource:2
$conc_biasadd_readvariableop_resource:>
$conb3_conv2d_readvariableop_resource:(3
%conb3_biasadd_readvariableop_resource:(>
$cona3_conv2d_readvariableop_resource:(3
%cona3_biasadd_readvariableop_resource:(7
#hida_matmul_readvariableop_resource:
??3
$hida_biasadd_readvariableop_resource:	?8
$hida2_matmul_readvariableop_resource:
??4
%hida2_biasadd_readvariableop_resource:	?7
$hida3_matmul_readvariableop_resource:	(?4
%hida3_biasadd_readvariableop_resource:	?7
#hidb_matmul_readvariableop_resource:
??3
$hidb_biasadd_readvariableop_resource:	?8
$hidb2_matmul_readvariableop_resource:
??4
%hidb2_biasadd_readvariableop_resource:	?7
$hidb3_matmul_readvariableop_resource:	(?4
%hidb3_biasadd_readvariableop_resource:	?7
#hidc_matmul_readvariableop_resource:
??3
$hidc_biasadd_readvariableop_resource:	?8
$hidd1_matmul_readvariableop_resource:
??4
%hidd1_biasadd_readvariableop_resource:	?8
$hide1_matmul_readvariableop_resource:
??4
%hide1_biasadd_readvariableop_resource:	?7
#hid1_matmul_readvariableop_resource:
?$?3
$hid1_biasadd_readvariableop_resource:	?7
#hid2_matmul_readvariableop_resource:
??3
$hid2_biasadd_readvariableop_resource:	?7
#hid3_matmul_readvariableop_resource:
??3
$hid3_biasadd_readvariableop_resource:	?8
%output_matmul_readvariableop_resource:	?
4
&output_biasadd_readvariableop_resource:

identity??ConA/BiasAdd/ReadVariableOp?ConA/Conv2D/ReadVariableOp?ConA2/BiasAdd/ReadVariableOp?ConA2/Conv2D/ReadVariableOp?ConA3/BiasAdd/ReadVariableOp?ConA3/Conv2D/ReadVariableOp?ConB/BiasAdd/ReadVariableOp?ConB/Conv2D/ReadVariableOp?ConB2/BiasAdd/ReadVariableOp?ConB2/Conv2D/ReadVariableOp?ConB3/BiasAdd/ReadVariableOp?ConB3/Conv2D/ReadVariableOp?ConC/BiasAdd/ReadVariableOp?ConC/Conv2D/ReadVariableOp?ConD/BiasAdd/ReadVariableOp?ConD/Conv2D/ReadVariableOp?ConE/BiasAdd/ReadVariableOp?ConE/Conv2D/ReadVariableOp?Hid1/BiasAdd/ReadVariableOp?Hid1/MatMul/ReadVariableOp?Hid2/BiasAdd/ReadVariableOp?Hid2/MatMul/ReadVariableOp?Hid3/BiasAdd/ReadVariableOp?Hid3/MatMul/ReadVariableOp?HidA/BiasAdd/ReadVariableOp?HidA/MatMul/ReadVariableOp?HidA2/BiasAdd/ReadVariableOp?HidA2/MatMul/ReadVariableOp?HidA3/BiasAdd/ReadVariableOp?HidA3/MatMul/ReadVariableOp?HidB/BiasAdd/ReadVariableOp?HidB/MatMul/ReadVariableOp?HidB2/BiasAdd/ReadVariableOp?HidB2/MatMul/ReadVariableOp?HidB3/BiasAdd/ReadVariableOp?HidB3/MatMul/ReadVariableOp?HidC/BiasAdd/ReadVariableOp?HidC/MatMul/ReadVariableOp?HidD1/BiasAdd/ReadVariableOp?HidD1/MatMul/ReadVariableOp?HidE1/BiasAdd/ReadVariableOp?HidE1/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
ConB/Conv2D/ReadVariableOpReadVariableOp#conb_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConB/Conv2D/ReadVariableOp?
ConB/Conv2DConv2Dinputs"ConB/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
ConB/Conv2D?
ConB/BiasAdd/ReadVariableOpReadVariableOp$conb_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConB/BiasAdd/ReadVariableOp?
ConB/BiasAddBiasAddConB/Conv2D:output:0#ConB/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
ConB/BiasAddo
	ConB/ReluReluConB/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
	ConB/Relu?
ConA/Conv2D/ReadVariableOpReadVariableOp#cona_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConA/Conv2D/ReadVariableOp?
ConA/Conv2DConv2Dinputs"ConA/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
ConA/Conv2D?
ConA/BiasAdd/ReadVariableOpReadVariableOp$cona_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConA/BiasAdd/ReadVariableOp?
ConA/BiasAddBiasAddConA/Conv2D:output:0#ConA/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
ConA/BiasAddo
	ConA/ReluReluConA/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
	ConA/Relu?
PoolB/MaxPoolMaxPoolConB/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
PoolB/MaxPool?
PoolA/MaxPoolMaxPoolConA/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
PoolA/MaxPool?
dropout_1/IdentityIdentityPoolB/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
dropout_1/Identity?
dropout/IdentityIdentityPoolA/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
dropout/Identity?
ConB2/Conv2D/ReadVariableOpReadVariableOp$conb2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConB2/Conv2D/ReadVariableOp?
ConB2/Conv2DConv2Ddropout_1/Identity:output:0#ConB2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		*
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
:?????????		2
ConB2/BiasAddr

ConB2/ReluReluConB2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????		2

ConB2/Relu?
ConA2/Conv2D/ReadVariableOpReadVariableOp$cona2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConA2/Conv2D/ReadVariableOp?
ConA2/Conv2DConv2Ddropout/Identity:output:0#ConA2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
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
:?????????

2
ConA2/BiasAddr

ConA2/ReluReluConA2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

2

ConA2/Relu?
PoolB2/MaxPoolMaxPoolConB2/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
PoolB2/MaxPool?
PoolA2/MaxPoolMaxPoolConA2/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
PoolA2/MaxPool?
ConE/Conv2D/ReadVariableOpReadVariableOp#cone_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConE/Conv2D/ReadVariableOp?
ConE/Conv2DConv2Dinputs"ConE/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
ConE/Conv2D?
ConE/BiasAdd/ReadVariableOpReadVariableOp$cone_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConE/BiasAdd/ReadVariableOp?
ConE/BiasAddBiasAddConE/Conv2D:output:0#ConE/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
ConE/BiasAddo
	ConE/ReluReluConE/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
	ConE/Relu?
ConD/Conv2D/ReadVariableOpReadVariableOp#cond_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
ConD/Conv2D/ReadVariableOp?
ConD/Conv2DConv2Dinputs"ConD/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
ConD/Conv2D?
ConD/BiasAdd/ReadVariableOpReadVariableOp$cond_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConD/BiasAdd/ReadVariableOp?
ConD/BiasAddBiasAddConD/Conv2D:output:0#ConD/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
ConD/BiasAddo
	ConD/ReluReluConD/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
	ConD/Relu?
ConC/Conv2D/ReadVariableOpReadVariableOp#conc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
ConC/Conv2D/ReadVariableOp?
ConC/Conv2DConv2Dinputs"ConC/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
ConC/Conv2D?
ConC/BiasAdd/ReadVariableOpReadVariableOp$conc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
ConC/BiasAdd/ReadVariableOp?
ConC/BiasAddBiasAddConC/Conv2D:output:0#ConC/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
ConC/BiasAddo
	ConC/ReluReluConC/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
	ConC/Relu?
ConB3/Conv2D/ReadVariableOpReadVariableOp$conb3_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
ConB3/Conv2D/ReadVariableOp?
ConB3/Conv2DConv2DPoolB2/MaxPool:output:0#ConB3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2
ConB3/BiasAddr

ConB3/ReluReluConB3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(2

ConB3/Relu?
ConA3/Conv2D/ReadVariableOpReadVariableOp$cona3_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
ConA3/Conv2D/ReadVariableOp?
ConA3/Conv2DConv2DPoolA2/MaxPool:output:0#ConA3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2
ConA3/BiasAddr

ConA3/ReluReluConA3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(2

ConA3/Relu?
PoolE/MaxPoolMaxPoolConE/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
PoolE/MaxPool?
PoolD/MaxPoolMaxPoolConD/Relu:activations:0*/
_output_shapes
:?????????		*
ksize
*
paddingVALID*
strides
2
PoolD/MaxPool?
PoolC/MaxPoolMaxPoolConC/Relu:activations:0*/
_output_shapes
:?????????

*
ksize
*
paddingVALID*
strides
2
PoolC/MaxPool?
PoolB3/MaxPoolMaxPoolConB3/Relu:activations:0*/
_output_shapes
:?????????(*
ksize
*
paddingVALID*
strides
2
PoolB3/MaxPool?
PoolA3/MaxPoolMaxPoolConA3/Relu:activations:0*/
_output_shapes
:?????????(*
ksize
*
paddingVALID*
strides
2
PoolA3/MaxPool?
dropout_4/IdentityIdentityPoolE/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
dropout_4/Identity?
dropout_3/IdentityIdentityPoolD/MaxPool:output:0*
T0*/
_output_shapes
:?????????		2
dropout_3/Identity?
dropout_2/IdentityIdentityPoolC/MaxPool:output:0*
T0*/
_output_shapes
:?????????

2
dropout_2/Identity?
dropout_8/IdentityIdentityPoolB3/MaxPool:output:0*
T0*/
_output_shapes
:?????????(2
dropout_8/Identity?
dropout_7/IdentityIdentityPoolB2/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
dropout_7/Identity?
dropout_6/IdentityIdentityPoolA3/MaxPool:output:0*
T0*/
_output_shapes
:?????????(2
dropout_6/Identity?
dropout_5/IdentityIdentityPoolA2/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
dropout_5/Identitys
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????`  2
flatten_8/Const?
flatten_8/ReshapeReshapedropout_4/Identity:output:0flatten_8/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_8/Reshapes
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_7/Const?
flatten_7/ReshapeReshapedropout_3/Identity:output:0flatten_7/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_7/Reshapes
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_6/Const?
flatten_6/ReshapeReshapedropout_2/Identity:output:0flatten_6/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_6/Reshapes
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????(   2
flatten_5/Const?
flatten_5/ReshapeReshapedropout_8/Identity:output:0flatten_5/Const:output:0*
T0*'
_output_shapes
:?????????(2
flatten_5/Reshapes
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
flatten_4/Const?
flatten_4/ReshapeReshapedropout_7/Identity:output:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_4/Reshapes
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_3/Const?
flatten_3/ReshapeReshapePoolB/MaxPool:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_3/Reshapes
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????(   2
flatten_2/Const?
flatten_2/ReshapeReshapedropout_6/Identity:output:0flatten_2/Const:output:0*
T0*'
_output_shapes
:?????????(2
flatten_2/Reshapes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_1/Const?
flatten_1/ReshapeReshapedropout_5/Identity:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_1/Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????`  2
flatten/Const?
flatten/ReshapeReshapePoolA/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
HidA/MatMul/ReadVariableOpReadVariableOp#hida_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
HidA/MatMul/ReadVariableOp?
HidA/MatMulMatMulflatten/Reshape:output:0"HidA/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA/MatMul?
HidA/BiasAdd/ReadVariableOpReadVariableOp$hida_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidA/BiasAdd/ReadVariableOp?
HidA/BiasAddBiasAddHidA/MatMul:product:0#HidA/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA/BiasAddh
	HidA/ReluReluHidA/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
	HidA/Relu?
HidA2/MatMul/ReadVariableOpReadVariableOp$hida2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
HidA2/MatMul/ReadVariableOp?
HidA2/MatMulMatMulflatten_1/Reshape:output:0#HidA2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA2/MatMul?
HidA2/BiasAdd/ReadVariableOpReadVariableOp%hida2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidA2/BiasAdd/ReadVariableOp?
HidA2/BiasAddBiasAddHidA2/MatMul:product:0$HidA2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA2/BiasAddk

HidA2/ReluReluHidA2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidA2/Relu?
HidA3/MatMul/ReadVariableOpReadVariableOp$hida3_matmul_readvariableop_resource*
_output_shapes
:	(?*
dtype02
HidA3/MatMul/ReadVariableOp?
HidA3/MatMulMatMulflatten_2/Reshape:output:0#HidA3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA3/MatMul?
HidA3/BiasAdd/ReadVariableOpReadVariableOp%hida3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidA3/BiasAdd/ReadVariableOp?
HidA3/BiasAddBiasAddHidA3/MatMul:product:0$HidA3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidA3/BiasAddk

HidA3/ReluReluHidA3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidA3/Relu?
HidB/MatMul/ReadVariableOpReadVariableOp#hidb_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
HidB/MatMul/ReadVariableOp?
HidB/MatMulMatMulflatten_3/Reshape:output:0"HidB/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidB/MatMul?
HidB/BiasAdd/ReadVariableOpReadVariableOp$hidb_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidB/BiasAdd/ReadVariableOp?
HidB/BiasAddBiasAddHidB/MatMul:product:0#HidB/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidB/BiasAddh
	HidB/ReluReluHidB/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
	HidB/Relu?
HidB2/MatMul/ReadVariableOpReadVariableOp$hidb2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
HidB2/MatMul/ReadVariableOp?
HidB2/MatMulMatMulflatten_4/Reshape:output:0#HidB2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidB2/MatMul?
HidB2/BiasAdd/ReadVariableOpReadVariableOp%hidb2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidB2/BiasAdd/ReadVariableOp?
HidB2/BiasAddBiasAddHidB2/MatMul:product:0$HidB2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidB2/BiasAddk

HidB2/ReluReluHidB2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidB2/Relu?
HidB3/MatMul/ReadVariableOpReadVariableOp$hidb3_matmul_readvariableop_resource*
_output_shapes
:	(?*
dtype02
HidB3/MatMul/ReadVariableOp?
HidB3/MatMulMatMulflatten_5/Reshape:output:0#HidB3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidB3/MatMul?
HidB3/BiasAdd/ReadVariableOpReadVariableOp%hidb3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidB3/BiasAdd/ReadVariableOp?
HidB3/BiasAddBiasAddHidB3/MatMul:product:0$HidB3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidB3/BiasAddk

HidB3/ReluReluHidB3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidB3/Relu?
HidC/MatMul/ReadVariableOpReadVariableOp#hidc_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
HidC/MatMul/ReadVariableOp?
HidC/MatMulMatMulflatten_6/Reshape:output:0"HidC/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidC/MatMul?
HidC/BiasAdd/ReadVariableOpReadVariableOp$hidc_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidC/BiasAdd/ReadVariableOp?
HidC/BiasAddBiasAddHidC/MatMul:product:0#HidC/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidC/BiasAddh
	HidC/ReluReluHidC/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
	HidC/Relu?
HidD1/MatMul/ReadVariableOpReadVariableOp$hidd1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
HidD1/MatMul/ReadVariableOp?
HidD1/MatMulMatMulflatten_7/Reshape:output:0#HidD1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidD1/MatMul?
HidD1/BiasAdd/ReadVariableOpReadVariableOp%hidd1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidD1/BiasAdd/ReadVariableOp?
HidD1/BiasAddBiasAddHidD1/MatMul:product:0$HidD1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidD1/BiasAddk

HidD1/ReluReluHidD1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidD1/Relu?
HidE1/MatMul/ReadVariableOpReadVariableOp$hide1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
HidE1/MatMul/ReadVariableOp?
HidE1/MatMulMatMulflatten_8/Reshape:output:0#HidE1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidE1/MatMul?
HidE1/BiasAdd/ReadVariableOpReadVariableOp%hide1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
HidE1/BiasAdd/ReadVariableOp?
HidE1/BiasAddBiasAddHidE1/MatMul:product:0$HidE1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
HidE1/BiasAddk

HidE1/ReluReluHidE1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

HidE1/Relut
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2HidA/Relu:activations:0HidA2/Relu:activations:0HidA3/Relu:activations:0HidB/Relu:activations:0HidB2/Relu:activations:0HidB3/Relu:activations:0HidC/Relu:activations:0HidD1/Relu:activations:0HidE1/Relu:activations:0 concatenate/concat/axis:output:0*
N	*
T0*(
_output_shapes
:??????????$2
concatenate/concat?
Hid1/MatMul/ReadVariableOpReadVariableOp#hid1_matmul_readvariableop_resource* 
_output_shapes
:
?$?*
dtype02
Hid1/MatMul/ReadVariableOp?
Hid1/MatMulMatMulconcatenate/concat:output:0"Hid1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Hid1/MatMul?
Hid1/BiasAdd/ReadVariableOpReadVariableOp$hid1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
Hid1/BiasAdd/ReadVariableOp?
Hid1/BiasAddBiasAddHid1/MatMul:product:0#Hid1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Hid1/BiasAddh
	Hid1/ReluReluHid1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
	Hid1/Relu?
Hid2/MatMul/ReadVariableOpReadVariableOp#hid2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Hid2/MatMul/ReadVariableOp?
Hid2/MatMulMatMulHid1/Relu:activations:0"Hid2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Hid2/MatMul?
Hid2/BiasAdd/ReadVariableOpReadVariableOp$hid2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
Hid2/BiasAdd/ReadVariableOp?
Hid2/BiasAddBiasAddHid2/MatMul:product:0#Hid2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Hid2/BiasAddh
	Hid2/ReluReluHid2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
	Hid2/Relu?
Hid3/MatMul/ReadVariableOpReadVariableOp#hid3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Hid3/MatMul/ReadVariableOp?
Hid3/MatMulMatMulHid2/Relu:activations:0"Hid3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Hid3/MatMul?
Hid3/BiasAdd/ReadVariableOpReadVariableOp$hid3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
Hid3/BiasAdd/ReadVariableOp?
Hid3/BiasAddBiasAddHid3/MatMul:product:0#Hid3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Hid3/BiasAddh
	Hid3/TanhTanhHid3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
	Hid3/Tanh?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMulHid3/Tanh:y:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
output/Softmaxs
IdentityIdentityoutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?

NoOpNoOp^ConA/BiasAdd/ReadVariableOp^ConA/Conv2D/ReadVariableOp^ConA2/BiasAdd/ReadVariableOp^ConA2/Conv2D/ReadVariableOp^ConA3/BiasAdd/ReadVariableOp^ConA3/Conv2D/ReadVariableOp^ConB/BiasAdd/ReadVariableOp^ConB/Conv2D/ReadVariableOp^ConB2/BiasAdd/ReadVariableOp^ConB2/Conv2D/ReadVariableOp^ConB3/BiasAdd/ReadVariableOp^ConB3/Conv2D/ReadVariableOp^ConC/BiasAdd/ReadVariableOp^ConC/Conv2D/ReadVariableOp^ConD/BiasAdd/ReadVariableOp^ConD/Conv2D/ReadVariableOp^ConE/BiasAdd/ReadVariableOp^ConE/Conv2D/ReadVariableOp^Hid1/BiasAdd/ReadVariableOp^Hid1/MatMul/ReadVariableOp^Hid2/BiasAdd/ReadVariableOp^Hid2/MatMul/ReadVariableOp^Hid3/BiasAdd/ReadVariableOp^Hid3/MatMul/ReadVariableOp^HidA/BiasAdd/ReadVariableOp^HidA/MatMul/ReadVariableOp^HidA2/BiasAdd/ReadVariableOp^HidA2/MatMul/ReadVariableOp^HidA3/BiasAdd/ReadVariableOp^HidA3/MatMul/ReadVariableOp^HidB/BiasAdd/ReadVariableOp^HidB/MatMul/ReadVariableOp^HidB2/BiasAdd/ReadVariableOp^HidB2/MatMul/ReadVariableOp^HidB3/BiasAdd/ReadVariableOp^HidB3/MatMul/ReadVariableOp^HidC/BiasAdd/ReadVariableOp^HidC/MatMul/ReadVariableOp^HidD1/BiasAdd/ReadVariableOp^HidD1/MatMul/ReadVariableOp^HidE1/BiasAdd/ReadVariableOp^HidE1/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
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
ConC/Conv2D/ReadVariableOpConC/Conv2D/ReadVariableOp2:
ConD/BiasAdd/ReadVariableOpConD/BiasAdd/ReadVariableOp28
ConD/Conv2D/ReadVariableOpConD/Conv2D/ReadVariableOp2:
ConE/BiasAdd/ReadVariableOpConE/BiasAdd/ReadVariableOp28
ConE/Conv2D/ReadVariableOpConE/Conv2D/ReadVariableOp2:
Hid1/BiasAdd/ReadVariableOpHid1/BiasAdd/ReadVariableOp28
Hid1/MatMul/ReadVariableOpHid1/MatMul/ReadVariableOp2:
Hid2/BiasAdd/ReadVariableOpHid2/BiasAdd/ReadVariableOp28
Hid2/MatMul/ReadVariableOpHid2/MatMul/ReadVariableOp2:
Hid3/BiasAdd/ReadVariableOpHid3/BiasAdd/ReadVariableOp28
Hid3/MatMul/ReadVariableOpHid3/MatMul/ReadVariableOp2:
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
HidD1/BiasAdd/ReadVariableOpHidD1/BiasAdd/ReadVariableOp2:
HidD1/MatMul/ReadVariableOpHidD1/MatMul/ReadVariableOp2<
HidE1/BiasAdd/ReadVariableOpHidE1/BiasAdd/ReadVariableOp2:
HidE1/MatMul/ReadVariableOpHidE1/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
@__inference_ConB2_layer_call_and_return_conditional_losses_20599

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		*
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
:?????????		2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????		2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????		2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_7_layer_call_fn_23664

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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_214632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
?__inference_Hid3_layer_call_and_return_conditional_losses_24155

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
@__inference_ConB3_layer_call_and_return_conditional_losses_23440

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
:?????????(*
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
:?????????(2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????(2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
@__inference_HidE1_layer_call_and_return_conditional_losses_21017

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
?__inference_Hid3_layer_call_and_return_conditional_losses_21084

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
?
?
%__inference_model_layer_call_fn_22638

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:		

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:(

unknown_14:($

unknown_15:(

unknown_16:(

unknown_17:
??

unknown_18:	?

unknown_19:
??

unknown_20:	?

unknown_21:	(?

unknown_22:	?

unknown_23:
??

unknown_24:	?

unknown_25:
??

unknown_26:	?

unknown_27:	(?

unknown_28:	?

unknown_29:
??

unknown_30:	?

unknown_31:
??

unknown_32:	?

unknown_33:
??

unknown_34:	?

unknown_35:
?$?

unknown_36:	?

unknown_37:
??

unknown_38:	?

unknown_39:
??

unknown_40:	?

unknown_41:	?


unknown_42:

identity??StatefulPartitionedCall?
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
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_211082
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$__inference_HidA_layer_call_fn_23897

inputs
unknown:
??
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
GPU2*0J 8? *H
fCRA
?__inference_HidA_layer_call_and_return_conditional_losses_208812
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
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_5_layer_call_and_return_conditional_losses_23627

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
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
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
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
\
@__inference_PoolB_layer_call_and_return_conditional_losses_20354

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
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
?__inference_ConC_layer_call_and_return_conditional_losses_20679

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
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
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
]
A__inference_PoolB2_layer_call_and_return_conditional_losses_20398

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
??
?
@__inference_model_layer_call_and_return_conditional_losses_22302	
input$

conb_22163:

conb_22165:$

cona_22168:

cona_22170:%
conb2_22177:
conb2_22179:%
cona2_22182:
cona2_22184:$

cone_22189:

cone_22191:$

cond_22194:		

cond_22196:$

conc_22199:

conc_22201:%
conb3_22204:(
conb3_22206:(%
cona3_22209:(
cona3_22211:(

hida_22235:
??

hida_22237:	?
hida2_22240:
??
hida2_22242:	?
hida3_22245:	(?
hida3_22247:	?

hidb_22250:
??

hidb_22252:	?
hidb2_22255:
??
hidb2_22257:	?
hidb3_22260:	(?
hidb3_22262:	?

hidc_22265:
??

hidc_22267:	?
hidd1_22270:
??
hidd1_22272:	?
hide1_22275:
??
hide1_22277:	?

hid1_22281:
?$?

hid1_22283:	?

hid2_22286:
??

hid2_22288:	?

hid3_22291:
??

hid3_22293:	?
output_22296:	?

output_22298:

identity??ConA/StatefulPartitionedCall?ConA2/StatefulPartitionedCall?ConA3/StatefulPartitionedCall?ConB/StatefulPartitionedCall?ConB2/StatefulPartitionedCall?ConB3/StatefulPartitionedCall?ConC/StatefulPartitionedCall?ConD/StatefulPartitionedCall?ConE/StatefulPartitionedCall?Hid1/StatefulPartitionedCall?Hid2/StatefulPartitionedCall?Hid3/StatefulPartitionedCall?HidA/StatefulPartitionedCall?HidA2/StatefulPartitionedCall?HidA3/StatefulPartitionedCall?HidB/StatefulPartitionedCall?HidB2/StatefulPartitionedCall?HidB3/StatefulPartitionedCall?HidC/StatefulPartitionedCall?HidD1/StatefulPartitionedCall?HidE1/StatefulPartitionedCall?output/StatefulPartitionedCall?
ConB/StatefulPartitionedCallStatefulPartitionedCallinput
conb_22163
conb_22165*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_ConB_layer_call_and_return_conditional_losses_205392
ConB/StatefulPartitionedCall?
ConA/StatefulPartitionedCallStatefulPartitionedCallinput
cona_22168
cona_22170*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_ConA_layer_call_and_return_conditional_losses_205562
ConA/StatefulPartitionedCall?
PoolB/PartitionedCallPartitionedCall%ConB/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_PoolB_layer_call_and_return_conditional_losses_205662
PoolB/PartitionedCall?
PoolA/PartitionedCallPartitionedCall%ConA/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_PoolA_layer_call_and_return_conditional_losses_205722
PoolA/PartitionedCall?
dropout_1/PartitionedCallPartitionedCallPoolB/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_205792
dropout_1/PartitionedCall?
dropout/PartitionedCallPartitionedCallPoolA/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_205862
dropout/PartitionedCall?
ConB2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conb2_22177conb2_22179*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConB2_layer_call_and_return_conditional_losses_205992
ConB2/StatefulPartitionedCall?
ConA2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0cona2_22182cona2_22184*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConA2_layer_call_and_return_conditional_losses_206162
ConA2/StatefulPartitionedCall?
PoolB2/PartitionedCallPartitionedCall&ConB2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolB2_layer_call_and_return_conditional_losses_206262
PoolB2/PartitionedCall?
PoolA2/PartitionedCallPartitionedCall&ConA2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolA2_layer_call_and_return_conditional_losses_206322
PoolA2/PartitionedCall?
ConE/StatefulPartitionedCallStatefulPartitionedCallinput
cone_22189
cone_22191*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_ConE_layer_call_and_return_conditional_losses_206452
ConE/StatefulPartitionedCall?
ConD/StatefulPartitionedCallStatefulPartitionedCallinput
cond_22194
cond_22196*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_ConD_layer_call_and_return_conditional_losses_206622
ConD/StatefulPartitionedCall?
ConC/StatefulPartitionedCallStatefulPartitionedCallinput
conc_22199
conc_22201*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_ConC_layer_call_and_return_conditional_losses_206792
ConC/StatefulPartitionedCall?
ConB3/StatefulPartitionedCallStatefulPartitionedCallPoolB2/PartitionedCall:output:0conb3_22204conb3_22206*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConB3_layer_call_and_return_conditional_losses_206962
ConB3/StatefulPartitionedCall?
ConA3/StatefulPartitionedCallStatefulPartitionedCallPoolA2/PartitionedCall:output:0cona3_22209cona3_22211*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_ConA3_layer_call_and_return_conditional_losses_207132
ConA3/StatefulPartitionedCall?
PoolE/PartitionedCallPartitionedCall%ConE/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_PoolE_layer_call_and_return_conditional_losses_207232
PoolE/PartitionedCall?
PoolD/PartitionedCallPartitionedCall%ConD/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_PoolD_layer_call_and_return_conditional_losses_207292
PoolD/PartitionedCall?
PoolC/PartitionedCallPartitionedCall%ConC/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_PoolC_layer_call_and_return_conditional_losses_207352
PoolC/PartitionedCall?
PoolB3/PartitionedCallPartitionedCall&ConB3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolB3_layer_call_and_return_conditional_losses_207412
PoolB3/PartitionedCall?
PoolA3/PartitionedCallPartitionedCall&ConA3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_PoolA3_layer_call_and_return_conditional_losses_207472
PoolA3/PartitionedCall?
dropout_4/PartitionedCallPartitionedCallPoolE/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_207542
dropout_4/PartitionedCall?
dropout_3/PartitionedCallPartitionedCallPoolD/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_207612
dropout_3/PartitionedCall?
dropout_2/PartitionedCallPartitionedCallPoolC/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_207682
dropout_2/PartitionedCall?
dropout_8/PartitionedCallPartitionedCallPoolB3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_207752
dropout_8/PartitionedCall?
dropout_7/PartitionedCallPartitionedCallPoolB2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_207822
dropout_7/PartitionedCall?
dropout_6/PartitionedCallPartitionedCallPoolA3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_207892
dropout_6/PartitionedCall?
dropout_5/PartitionedCallPartitionedCallPoolA2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_207962
dropout_5/PartitionedCall?
flatten_8/PartitionedCallPartitionedCall"dropout_4/PartitionedCall:output:0*
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
GPU2*0J 8? *M
fHRF
D__inference_flatten_8_layer_call_and_return_conditional_losses_208042
flatten_8/PartitionedCall?
flatten_7/PartitionedCallPartitionedCall"dropout_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_7_layer_call_and_return_conditional_losses_208122
flatten_7/PartitionedCall?
flatten_6/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_208202
flatten_6/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall"dropout_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_208282
flatten_5/PartitionedCall?
flatten_4/PartitionedCallPartitionedCall"dropout_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_208362
flatten_4/PartitionedCall?
flatten_3/PartitionedCallPartitionedCallPoolB/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_208442
flatten_3/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall"dropout_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_208522
flatten_2/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall"dropout_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_208602
flatten_1/PartitionedCall?
flatten/PartitionedCallPartitionedCallPoolA/PartitionedCall:output:0*
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
GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_208682
flatten/PartitionedCall?
HidA/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
hida_22235
hida_22237*
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
GPU2*0J 8? *H
fCRA
?__inference_HidA_layer_call_and_return_conditional_losses_208812
HidA/StatefulPartitionedCall?
HidA2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0hida2_22240hida2_22242*
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
@__inference_HidA2_layer_call_and_return_conditional_losses_208982
HidA2/StatefulPartitionedCall?
HidA3/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0hida3_22245hida3_22247*
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
@__inference_HidA3_layer_call_and_return_conditional_losses_209152
HidA3/StatefulPartitionedCall?
HidB/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0
hidb_22250
hidb_22252*
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
GPU2*0J 8? *H
fCRA
?__inference_HidB_layer_call_and_return_conditional_losses_209322
HidB/StatefulPartitionedCall?
HidB2/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0hidb2_22255hidb2_22257*
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
@__inference_HidB2_layer_call_and_return_conditional_losses_209492
HidB2/StatefulPartitionedCall?
HidB3/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0hidb3_22260hidb3_22262*
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
@__inference_HidB3_layer_call_and_return_conditional_losses_209662
HidB3/StatefulPartitionedCall?
HidC/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0
hidc_22265
hidc_22267*
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
GPU2*0J 8? *H
fCRA
?__inference_HidC_layer_call_and_return_conditional_losses_209832
HidC/StatefulPartitionedCall?
HidD1/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0hidd1_22270hidd1_22272*
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
@__inference_HidD1_layer_call_and_return_conditional_losses_210002
HidD1/StatefulPartitionedCall?
HidE1/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0hide1_22275hide1_22277*
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
@__inference_HidE1_layer_call_and_return_conditional_losses_210172
HidE1/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall%HidA/StatefulPartitionedCall:output:0&HidA2/StatefulPartitionedCall:output:0&HidA3/StatefulPartitionedCall:output:0%HidB/StatefulPartitionedCall:output:0&HidB2/StatefulPartitionedCall:output:0&HidB3/StatefulPartitionedCall:output:0%HidC/StatefulPartitionedCall:output:0&HidD1/StatefulPartitionedCall:output:0&HidE1/StatefulPartitionedCall:output:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_210372
concatenate/PartitionedCall?
Hid1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0
hid1_22281
hid1_22283*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_Hid1_layer_call_and_return_conditional_losses_210502
Hid1/StatefulPartitionedCall?
Hid2/StatefulPartitionedCallStatefulPartitionedCall%Hid1/StatefulPartitionedCall:output:0
hid2_22286
hid2_22288*
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
GPU2*0J 8? *H
fCRA
?__inference_Hid2_layer_call_and_return_conditional_losses_210672
Hid2/StatefulPartitionedCall?
Hid3/StatefulPartitionedCallStatefulPartitionedCall%Hid2/StatefulPartitionedCall:output:0
hid3_22291
hid3_22293*
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
GPU2*0J 8? *H
fCRA
?__inference_Hid3_layer_call_and_return_conditional_losses_210842
Hid3/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall%Hid3/StatefulPartitionedCall:output:0output_22296output_22298*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_211012 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp^ConA/StatefulPartitionedCall^ConA2/StatefulPartitionedCall^ConA3/StatefulPartitionedCall^ConB/StatefulPartitionedCall^ConB2/StatefulPartitionedCall^ConB3/StatefulPartitionedCall^ConC/StatefulPartitionedCall^ConD/StatefulPartitionedCall^ConE/StatefulPartitionedCall^Hid1/StatefulPartitionedCall^Hid2/StatefulPartitionedCall^Hid3/StatefulPartitionedCall^HidA/StatefulPartitionedCall^HidA2/StatefulPartitionedCall^HidA3/StatefulPartitionedCall^HidB/StatefulPartitionedCall^HidB2/StatefulPartitionedCall^HidB3/StatefulPartitionedCall^HidC/StatefulPartitionedCall^HidD1/StatefulPartitionedCall^HidE1/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesu
s:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
ConA/StatefulPartitionedCallConA/StatefulPartitionedCall2>
ConA2/StatefulPartitionedCallConA2/StatefulPartitionedCall2>
ConA3/StatefulPartitionedCallConA3/StatefulPartitionedCall2<
ConB/StatefulPartitionedCallConB/StatefulPartitionedCall2>
ConB2/StatefulPartitionedCallConB2/StatefulPartitionedCall2>
ConB3/StatefulPartitionedCallConB3/StatefulPartitionedCall2<
ConC/StatefulPartitionedCallConC/StatefulPartitionedCall2<
ConD/StatefulPartitionedCallConD/StatefulPartitionedCall2<
ConE/StatefulPartitionedCallConE/StatefulPartitionedCall2<
Hid1/StatefulPartitionedCallHid1/StatefulPartitionedCall2<
Hid2/StatefulPartitionedCallHid2/StatefulPartitionedCall2<
Hid3/StatefulPartitionedCallHid3/StatefulPartitionedCall2<
HidA/StatefulPartitionedCallHidA/StatefulPartitionedCall2>
HidA2/StatefulPartitionedCallHidA2/StatefulPartitionedCall2>
HidA3/StatefulPartitionedCallHidA3/StatefulPartitionedCall2<
HidB/StatefulPartitionedCallHidB/StatefulPartitionedCall2>
HidB2/StatefulPartitionedCallHidB2/StatefulPartitionedCall2>
HidB3/StatefulPartitionedCallHidB3/StatefulPartitionedCall2<
HidC/StatefulPartitionedCallHidC/StatefulPartitionedCall2>
HidD1/StatefulPartitionedCallHidD1/StatefulPartitionedCall2>
HidE1/StatefulPartitionedCallHidE1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
input6
serving_default_input:0?????????:
output0
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?
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

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer-16
layer-17
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
&layer_with_weights-9
&layer-37
'layer_with_weights-10
'layer-38
(layer_with_weights-11
(layer-39
)layer_with_weights-12
)layer-40
*layer_with_weights-13
*layer-41
+layer_with_weights-14
+layer-42
,layer_with_weights-15
,layer-43
-layer_with_weights-16
-layer-44
.layer_with_weights-17
.layer-45
/layer-46
0layer_with_weights-18
0layer-47
1layer_with_weights-19
1layer-48
2layer_with_weights-20
2layer-49
3layer_with_weights-21
3layer-50
4	optimizer
5	variables
6regularization_losses
7trainable_variables
8	keras_api
9
signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?

:kernel
;bias
<	variables
=regularization_losses
>trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

@kernel
Abias
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Vkernel
Wbias
X	variables
Yregularization_losses
Ztrainable_variables
[	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

\kernel
]bias
^	variables
_regularization_losses
`trainable_variables
a	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
f	variables
gregularization_losses
htrainable_variables
i	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

jkernel
kbias
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

pkernel
qbias
r	variables
sregularization_losses
ttrainable_variables
u	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

vkernel
wbias
x	variables
yregularization_losses
ztrainable_variables
{	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

|kernel
}bias
~	variables
regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate:m?;m?@m?Am?Vm?Wm?\m?]m?jm?km?pm?qm?vm?wm?|m?}m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?:v?;v?@v?Av?Vv?Wv?\v?]v?jv?kv?pv?qv?vv?wv?|v?}v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
?
:0
;1
@2
A3
V4
W5
\6
]7
j8
k9
p10
q11
v12
w13
|14
}15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
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
?43"
trackable_list_wrapper
 "
trackable_list_wrapper
?
:0
;1
@2
A3
V4
W5
\6
]7
j8
k9
p10
q11
v12
w13
|14
}15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
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
?43"
trackable_list_wrapper
?
5	variables
 ?layer_regularization_losses
?layer_metrics
6regularization_losses
?non_trainable_variables
?layers
?metrics
7trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
%:#2ConA/kernel
:2	ConA/bias
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
?
<	variables
 ?layer_regularization_losses
?layer_metrics
=regularization_losses
?non_trainable_variables
?layers
?metrics
>trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#2ConB/kernel
:2	ConB/bias
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
?
B	variables
 ?layer_regularization_losses
?layer_metrics
Cregularization_losses
?non_trainable_variables
?layers
?metrics
Dtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
F	variables
 ?layer_regularization_losses
?layer_metrics
Gregularization_losses
?non_trainable_variables
?layers
?metrics
Htrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
J	variables
 ?layer_regularization_losses
?layer_metrics
Kregularization_losses
?non_trainable_variables
?layers
?metrics
Ltrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
N	variables
 ?layer_regularization_losses
?layer_metrics
Oregularization_losses
?non_trainable_variables
?layers
?metrics
Ptrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
R	variables
 ?layer_regularization_losses
?layer_metrics
Sregularization_losses
?non_trainable_variables
?layers
?metrics
Ttrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$2ConA2/kernel
:2
ConA2/bias
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
?
X	variables
 ?layer_regularization_losses
?layer_metrics
Yregularization_losses
?non_trainable_variables
?layers
?metrics
Ztrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$2ConB2/kernel
:2
ConB2/bias
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
?
^	variables
 ?layer_regularization_losses
?layer_metrics
_regularization_losses
?non_trainable_variables
?layers
?metrics
`trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
b	variables
 ?layer_regularization_losses
?layer_metrics
cregularization_losses
?non_trainable_variables
?layers
?metrics
dtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
f	variables
 ?layer_regularization_losses
?layer_metrics
gregularization_losses
?non_trainable_variables
?layers
?metrics
htrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$(2ConA3/kernel
:(2
ConA3/bias
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
?
l	variables
 ?layer_regularization_losses
?layer_metrics
mregularization_losses
?non_trainable_variables
?layers
?metrics
ntrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$(2ConB3/kernel
:(2
ConB3/bias
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
?
r	variables
 ?layer_regularization_losses
?layer_metrics
sregularization_losses
?non_trainable_variables
?layers
?metrics
ttrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#2ConC/kernel
:2	ConC/bias
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
?
x	variables
 ?layer_regularization_losses
?layer_metrics
yregularization_losses
?non_trainable_variables
?layers
?metrics
ztrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#		2ConD/kernel
:2	ConD/bias
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
?
~	variables
 ?layer_regularization_losses
?layer_metrics
regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#2ConE/kernel
:2	ConE/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
??2HidA/kernel
:?2	HidA/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :
??2HidA2/kernel
:?2
HidA2/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	(?2HidA3/kernel
:?2
HidA3/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
??2HidB/kernel
:?2	HidB/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :
??2HidB2/kernel
:?2
HidB2/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	(?2HidB3/kernel
:?2
HidB3/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
??2HidC/kernel
:?2	HidC/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :
??2HidD1/kernel
:?2
HidD1/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :
??2HidE1/kernel
:?2
HidE1/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
?$?2Hid1/kernel
:?2	Hid1/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
??2Hid2/kernel
:?2	Hid2/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
??2Hid3/kernel
:?2	Hid3/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :	?
2output/kernel
:
2output/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?layer_metrics
?regularization_losses
?non_trainable_variables
?layers
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
 "
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
350"
trackable_list_wrapper
0
?0
?1"
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

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
*:(2Adam/ConA/kernel/m
:2Adam/ConA/bias/m
*:(2Adam/ConB/kernel/m
:2Adam/ConB/bias/m
+:)2Adam/ConA2/kernel/m
:2Adam/ConA2/bias/m
+:)2Adam/ConB2/kernel/m
:2Adam/ConB2/bias/m
+:)(2Adam/ConA3/kernel/m
:(2Adam/ConA3/bias/m
+:)(2Adam/ConB3/kernel/m
:(2Adam/ConB3/bias/m
*:(2Adam/ConC/kernel/m
:2Adam/ConC/bias/m
*:(		2Adam/ConD/kernel/m
:2Adam/ConD/bias/m
*:(2Adam/ConE/kernel/m
:2Adam/ConE/bias/m
$:"
??2Adam/HidA/kernel/m
:?2Adam/HidA/bias/m
%:#
??2Adam/HidA2/kernel/m
:?2Adam/HidA2/bias/m
$:"	(?2Adam/HidA3/kernel/m
:?2Adam/HidA3/bias/m
$:"
??2Adam/HidB/kernel/m
:?2Adam/HidB/bias/m
%:#
??2Adam/HidB2/kernel/m
:?2Adam/HidB2/bias/m
$:"	(?2Adam/HidB3/kernel/m
:?2Adam/HidB3/bias/m
$:"
??2Adam/HidC/kernel/m
:?2Adam/HidC/bias/m
%:#
??2Adam/HidD1/kernel/m
:?2Adam/HidD1/bias/m
%:#
??2Adam/HidE1/kernel/m
:?2Adam/HidE1/bias/m
$:"
?$?2Adam/Hid1/kernel/m
:?2Adam/Hid1/bias/m
$:"
??2Adam/Hid2/kernel/m
:?2Adam/Hid2/bias/m
$:"
??2Adam/Hid3/kernel/m
:?2Adam/Hid3/bias/m
%:#	?
2Adam/output/kernel/m
:
2Adam/output/bias/m
*:(2Adam/ConA/kernel/v
:2Adam/ConA/bias/v
*:(2Adam/ConB/kernel/v
:2Adam/ConB/bias/v
+:)2Adam/ConA2/kernel/v
:2Adam/ConA2/bias/v
+:)2Adam/ConB2/kernel/v
:2Adam/ConB2/bias/v
+:)(2Adam/ConA3/kernel/v
:(2Adam/ConA3/bias/v
+:)(2Adam/ConB3/kernel/v
:(2Adam/ConB3/bias/v
*:(2Adam/ConC/kernel/v
:2Adam/ConC/bias/v
*:(		2Adam/ConD/kernel/v
:2Adam/ConD/bias/v
*:(2Adam/ConE/kernel/v
:2Adam/ConE/bias/v
$:"
??2Adam/HidA/kernel/v
:?2Adam/HidA/bias/v
%:#
??2Adam/HidA2/kernel/v
:?2Adam/HidA2/bias/v
$:"	(?2Adam/HidA3/kernel/v
:?2Adam/HidA3/bias/v
$:"
??2Adam/HidB/kernel/v
:?2Adam/HidB/bias/v
%:#
??2Adam/HidB2/kernel/v
:?2Adam/HidB2/bias/v
$:"	(?2Adam/HidB3/kernel/v
:?2Adam/HidB3/bias/v
$:"
??2Adam/HidC/kernel/v
:?2Adam/HidC/bias/v
%:#
??2Adam/HidD1/kernel/v
:?2Adam/HidD1/bias/v
%:#
??2Adam/HidE1/kernel/v
:?2Adam/HidE1/bias/v
$:"
?$?2Adam/Hid1/kernel/v
:?2Adam/Hid1/bias/v
$:"
??2Adam/Hid2/kernel/v
:?2Adam/Hid2/bias/v
$:"
??2Adam/Hid3/kernel/v
:?2Adam/Hid3/bias/v
%:#	?
2Adam/output/kernel/v
:
2Adam/output/bias/v
?2?
%__inference_model_layer_call_fn_21199
%__inference_model_layer_call_fn_22638
%__inference_model_layer_call_fn_22731
%__inference_model_layer_call_fn_22160?
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
@__inference_model_layer_call_and_return_conditional_losses_22927
@__inference_model_layer_call_and_return_conditional_losses_23186
@__inference_model_layer_call_and_return_conditional_losses_22302
@__inference_model_layer_call_and_return_conditional_losses_22444?
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
 __inference__wrapped_model_20323input"?
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
?2?
$__inference_ConA_layer_call_fn_23195?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_ConA_layer_call_and_return_conditional_losses_23206?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_ConB_layer_call_fn_23215?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_ConB_layer_call_and_return_conditional_losses_23226?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_PoolA_layer_call_fn_23231
%__inference_PoolA_layer_call_fn_23236?
???
FullArgSpec
args?
jself
jinputs
varargs
 
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
@__inference_PoolA_layer_call_and_return_conditional_losses_23241
@__inference_PoolA_layer_call_and_return_conditional_losses_23246?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_PoolB_layer_call_fn_23251
%__inference_PoolB_layer_call_fn_23256?
???
FullArgSpec
args?
jself
jinputs
varargs
 
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
@__inference_PoolB_layer_call_and_return_conditional_losses_23261
@__inference_PoolB_layer_call_and_return_conditional_losses_23266?
???
FullArgSpec
args?
jself
jinputs
varargs
 
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
'__inference_dropout_layer_call_fn_23271
'__inference_dropout_layer_call_fn_23276?
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
B__inference_dropout_layer_call_and_return_conditional_losses_23281
B__inference_dropout_layer_call_and_return_conditional_losses_23293?
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
)__inference_dropout_1_layer_call_fn_23298
)__inference_dropout_1_layer_call_fn_23303?
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
D__inference_dropout_1_layer_call_and_return_conditional_losses_23308
D__inference_dropout_1_layer_call_and_return_conditional_losses_23320?
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
%__inference_ConA2_layer_call_fn_23329?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_ConA2_layer_call_and_return_conditional_losses_23340?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_ConB2_layer_call_fn_23349?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_ConB2_layer_call_and_return_conditional_losses_23360?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_PoolA2_layer_call_fn_23365
&__inference_PoolA2_layer_call_fn_23370?
???
FullArgSpec
args?
jself
jinputs
varargs
 
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
A__inference_PoolA2_layer_call_and_return_conditional_losses_23375
A__inference_PoolA2_layer_call_and_return_conditional_losses_23380?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_PoolB2_layer_call_fn_23385
&__inference_PoolB2_layer_call_fn_23390?
???
FullArgSpec
args?
jself
jinputs
varargs
 
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
A__inference_PoolB2_layer_call_and_return_conditional_losses_23395
A__inference_PoolB2_layer_call_and_return_conditional_losses_23400?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_ConA3_layer_call_fn_23409?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_ConA3_layer_call_and_return_conditional_losses_23420?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_ConB3_layer_call_fn_23429?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_ConB3_layer_call_and_return_conditional_losses_23440?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_ConC_layer_call_fn_23449?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_ConC_layer_call_and_return_conditional_losses_23460?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_ConD_layer_call_fn_23469?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_ConD_layer_call_and_return_conditional_losses_23480?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_ConE_layer_call_fn_23489?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_ConE_layer_call_and_return_conditional_losses_23500?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_PoolA3_layer_call_fn_23505
&__inference_PoolA3_layer_call_fn_23510?
???
FullArgSpec
args?
jself
jinputs
varargs
 
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
A__inference_PoolA3_layer_call_and_return_conditional_losses_23515
A__inference_PoolA3_layer_call_and_return_conditional_losses_23520?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_PoolB3_layer_call_fn_23525
&__inference_PoolB3_layer_call_fn_23530?
???
FullArgSpec
args?
jself
jinputs
varargs
 
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
A__inference_PoolB3_layer_call_and_return_conditional_losses_23535
A__inference_PoolB3_layer_call_and_return_conditional_losses_23540?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_PoolC_layer_call_fn_23545
%__inference_PoolC_layer_call_fn_23550?
???
FullArgSpec
args?
jself
jinputs
varargs
 
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
@__inference_PoolC_layer_call_and_return_conditional_losses_23555
@__inference_PoolC_layer_call_and_return_conditional_losses_23560?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_PoolD_layer_call_fn_23565
%__inference_PoolD_layer_call_fn_23570?
???
FullArgSpec
args?
jself
jinputs
varargs
 
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
@__inference_PoolD_layer_call_and_return_conditional_losses_23575
@__inference_PoolD_layer_call_and_return_conditional_losses_23580?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_PoolE_layer_call_fn_23585
%__inference_PoolE_layer_call_fn_23590?
???
FullArgSpec
args?
jself
jinputs
varargs
 
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
@__inference_PoolE_layer_call_and_return_conditional_losses_23595
@__inference_PoolE_layer_call_and_return_conditional_losses_23600?
???
FullArgSpec
args?
jself
jinputs
varargs
 
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
)__inference_dropout_5_layer_call_fn_23605
)__inference_dropout_5_layer_call_fn_23610?
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
D__inference_dropout_5_layer_call_and_return_conditional_losses_23615
D__inference_dropout_5_layer_call_and_return_conditional_losses_23627?
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
)__inference_dropout_6_layer_call_fn_23632
)__inference_dropout_6_layer_call_fn_23637?
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
D__inference_dropout_6_layer_call_and_return_conditional_losses_23642
D__inference_dropout_6_layer_call_and_return_conditional_losses_23654?
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
)__inference_dropout_7_layer_call_fn_23659
)__inference_dropout_7_layer_call_fn_23664?
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
D__inference_dropout_7_layer_call_and_return_conditional_losses_23669
D__inference_dropout_7_layer_call_and_return_conditional_losses_23681?
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
)__inference_dropout_8_layer_call_fn_23686
)__inference_dropout_8_layer_call_fn_23691?
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
D__inference_dropout_8_layer_call_and_return_conditional_losses_23696
D__inference_dropout_8_layer_call_and_return_conditional_losses_23708?
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
)__inference_dropout_2_layer_call_fn_23713
)__inference_dropout_2_layer_call_fn_23718?
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
D__inference_dropout_2_layer_call_and_return_conditional_losses_23723
D__inference_dropout_2_layer_call_and_return_conditional_losses_23735?
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
)__inference_dropout_3_layer_call_fn_23740
)__inference_dropout_3_layer_call_fn_23745?
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
D__inference_dropout_3_layer_call_and_return_conditional_losses_23750
D__inference_dropout_3_layer_call_and_return_conditional_losses_23762?
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
)__inference_dropout_4_layer_call_fn_23767
)__inference_dropout_4_layer_call_fn_23772?
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
D__inference_dropout_4_layer_call_and_return_conditional_losses_23777
D__inference_dropout_4_layer_call_and_return_conditional_losses_23789?
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
'__inference_flatten_layer_call_fn_23794?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_flatten_layer_call_and_return_conditional_losses_23800?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_1_layer_call_fn_23805?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_1_layer_call_and_return_conditional_losses_23811?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_2_layer_call_fn_23816?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_2_layer_call_and_return_conditional_losses_23822?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_3_layer_call_fn_23827?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_3_layer_call_and_return_conditional_losses_23833?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_4_layer_call_fn_23838?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_4_layer_call_and_return_conditional_losses_23844?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_5_layer_call_fn_23849?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_5_layer_call_and_return_conditional_losses_23855?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_6_layer_call_fn_23860?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_6_layer_call_and_return_conditional_losses_23866?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_7_layer_call_fn_23871?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_7_layer_call_and_return_conditional_losses_23877?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_8_layer_call_fn_23882?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_8_layer_call_and_return_conditional_losses_23888?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_HidA_layer_call_fn_23897?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_HidA_layer_call_and_return_conditional_losses_23908?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_HidA2_layer_call_fn_23917?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_HidA2_layer_call_and_return_conditional_losses_23928?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_HidA3_layer_call_fn_23937?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_HidA3_layer_call_and_return_conditional_losses_23948?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_HidB_layer_call_fn_23957?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_HidB_layer_call_and_return_conditional_losses_23968?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_HidB2_layer_call_fn_23977?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_HidB2_layer_call_and_return_conditional_losses_23988?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_HidB3_layer_call_fn_23997?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_HidB3_layer_call_and_return_conditional_losses_24008?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_HidC_layer_call_fn_24017?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_HidC_layer_call_and_return_conditional_losses_24028?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_HidD1_layer_call_fn_24037?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_HidD1_layer_call_and_return_conditional_losses_24048?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_HidE1_layer_call_fn_24057?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_HidE1_layer_call_and_return_conditional_losses_24068?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_concatenate_layer_call_fn_24081?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_concatenate_layer_call_and_return_conditional_losses_24095?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_Hid1_layer_call_fn_24104?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_Hid1_layer_call_and_return_conditional_losses_24115?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_Hid2_layer_call_fn_24124?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_Hid2_layer_call_and_return_conditional_losses_24135?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_Hid3_layer_call_fn_24144?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_Hid3_layer_call_and_return_conditional_losses_24155?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_output_layer_call_fn_24164?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_output_layer_call_and_return_conditional_losses_24175?
???
FullArgSpec
args?
jself
jinputs
varargs
 
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
#__inference_signature_wrapper_22545input"?
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
@__inference_ConA2_layer_call_and_return_conditional_losses_23340lVW7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????


? ?
%__inference_ConA2_layer_call_fn_23329_VW7?4
-?*
(?%
inputs?????????
? " ??????????

?
@__inference_ConA3_layer_call_and_return_conditional_losses_23420ljk7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????(
? ?
%__inference_ConA3_layer_call_fn_23409_jk7?4
-?*
(?%
inputs?????????
? " ??????????(?
?__inference_ConA_layer_call_and_return_conditional_losses_23206l:;7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
$__inference_ConA_layer_call_fn_23195_:;7?4
-?*
(?%
inputs?????????
? " ???????????
@__inference_ConB2_layer_call_and_return_conditional_losses_23360l\]7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????		
? ?
%__inference_ConB2_layer_call_fn_23349_\]7?4
-?*
(?%
inputs?????????
? " ??????????		?
@__inference_ConB3_layer_call_and_return_conditional_losses_23440lpq7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????(
? ?
%__inference_ConB3_layer_call_fn_23429_pq7?4
-?*
(?%
inputs?????????
? " ??????????(?
?__inference_ConB_layer_call_and_return_conditional_losses_23226l@A7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
$__inference_ConB_layer_call_fn_23215_@A7?4
-?*
(?%
inputs?????????
? " ???????????
?__inference_ConC_layer_call_and_return_conditional_losses_23460lvw7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
$__inference_ConC_layer_call_fn_23449_vw7?4
-?*
(?%
inputs?????????
? " ???????????
?__inference_ConD_layer_call_and_return_conditional_losses_23480l|}7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
$__inference_ConD_layer_call_fn_23469_|}7?4
-?*
(?%
inputs?????????
? " ???????????
?__inference_ConE_layer_call_and_return_conditional_losses_23500n??7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
$__inference_ConE_layer_call_fn_23489a??7?4
-?*
(?%
inputs?????????
? " ???????????
?__inference_Hid1_layer_call_and_return_conditional_losses_24115`??0?-
&?#
!?
inputs??????????$
? "&?#
?
0??????????
? {
$__inference_Hid1_layer_call_fn_24104S??0?-
&?#
!?
inputs??????????$
? "????????????
?__inference_Hid2_layer_call_and_return_conditional_losses_24135`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
$__inference_Hid2_layer_call_fn_24124S??0?-
&?#
!?
inputs??????????
? "????????????
?__inference_Hid3_layer_call_and_return_conditional_losses_24155`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
$__inference_Hid3_layer_call_fn_24144S??0?-
&?#
!?
inputs??????????
? "????????????
@__inference_HidA2_layer_call_and_return_conditional_losses_23928`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? |
%__inference_HidA2_layer_call_fn_23917S??0?-
&?#
!?
inputs??????????
? "????????????
@__inference_HidA3_layer_call_and_return_conditional_losses_23948_??/?,
%?"
 ?
inputs?????????(
? "&?#
?
0??????????
? {
%__inference_HidA3_layer_call_fn_23937R??/?,
%?"
 ?
inputs?????????(
? "????????????
?__inference_HidA_layer_call_and_return_conditional_losses_23908`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
$__inference_HidA_layer_call_fn_23897S??0?-
&?#
!?
inputs??????????
? "????????????
@__inference_HidB2_layer_call_and_return_conditional_losses_23988`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? |
%__inference_HidB2_layer_call_fn_23977S??0?-
&?#
!?
inputs??????????
? "????????????
@__inference_HidB3_layer_call_and_return_conditional_losses_24008_??/?,
%?"
 ?
inputs?????????(
? "&?#
?
0??????????
? {
%__inference_HidB3_layer_call_fn_23997R??/?,
%?"
 ?
inputs?????????(
? "????????????
?__inference_HidB_layer_call_and_return_conditional_losses_23968`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
$__inference_HidB_layer_call_fn_23957S??0?-
&?#
!?
inputs??????????
? "????????????
?__inference_HidC_layer_call_and_return_conditional_losses_24028`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
$__inference_HidC_layer_call_fn_24017S??0?-
&?#
!?
inputs??????????
? "????????????
@__inference_HidD1_layer_call_and_return_conditional_losses_24048`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? |
%__inference_HidD1_layer_call_fn_24037S??0?-
&?#
!?
inputs??????????
? "????????????
@__inference_HidE1_layer_call_and_return_conditional_losses_24068`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? |
%__inference_HidE1_layer_call_fn_24057S??0?-
&?#
!?
inputs??????????
? "????????????
A__inference_PoolA2_layer_call_and_return_conditional_losses_23375?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
A__inference_PoolA2_layer_call_and_return_conditional_losses_23380h7?4
-?*
(?%
inputs?????????


? "-?*
#? 
0?????????
? ?
&__inference_PoolA2_layer_call_fn_23365?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
&__inference_PoolA2_layer_call_fn_23370[7?4
-?*
(?%
inputs?????????


? " ???????????
A__inference_PoolA3_layer_call_and_return_conditional_losses_23515?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
A__inference_PoolA3_layer_call_and_return_conditional_losses_23520h7?4
-?*
(?%
inputs?????????(
? "-?*
#? 
0?????????(
? ?
&__inference_PoolA3_layer_call_fn_23505?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
&__inference_PoolA3_layer_call_fn_23510[7?4
-?*
(?%
inputs?????????(
? " ??????????(?
@__inference_PoolA_layer_call_and_return_conditional_losses_23241?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
@__inference_PoolA_layer_call_and_return_conditional_losses_23246h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
%__inference_PoolA_layer_call_fn_23231?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
%__inference_PoolA_layer_call_fn_23236[7?4
-?*
(?%
inputs?????????
? " ???????????
A__inference_PoolB2_layer_call_and_return_conditional_losses_23395?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
A__inference_PoolB2_layer_call_and_return_conditional_losses_23400h7?4
-?*
(?%
inputs?????????		
? "-?*
#? 
0?????????
? ?
&__inference_PoolB2_layer_call_fn_23385?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
&__inference_PoolB2_layer_call_fn_23390[7?4
-?*
(?%
inputs?????????		
? " ???????????
A__inference_PoolB3_layer_call_and_return_conditional_losses_23535?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
A__inference_PoolB3_layer_call_and_return_conditional_losses_23540h7?4
-?*
(?%
inputs?????????(
? "-?*
#? 
0?????????(
? ?
&__inference_PoolB3_layer_call_fn_23525?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
&__inference_PoolB3_layer_call_fn_23530[7?4
-?*
(?%
inputs?????????(
? " ??????????(?
@__inference_PoolB_layer_call_and_return_conditional_losses_23261?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
@__inference_PoolB_layer_call_and_return_conditional_losses_23266h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
%__inference_PoolB_layer_call_fn_23251?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
%__inference_PoolB_layer_call_fn_23256[7?4
-?*
(?%
inputs?????????
? " ???????????
@__inference_PoolC_layer_call_and_return_conditional_losses_23555?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
@__inference_PoolC_layer_call_and_return_conditional_losses_23560h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????


? ?
%__inference_PoolC_layer_call_fn_23545?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
%__inference_PoolC_layer_call_fn_23550[7?4
-?*
(?%
inputs?????????
? " ??????????

?
@__inference_PoolD_layer_call_and_return_conditional_losses_23575?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
@__inference_PoolD_layer_call_and_return_conditional_losses_23580h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????		
? ?
%__inference_PoolD_layer_call_fn_23565?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
%__inference_PoolD_layer_call_fn_23570[7?4
-?*
(?%
inputs?????????
? " ??????????		?
@__inference_PoolE_layer_call_and_return_conditional_losses_23595?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
@__inference_PoolE_layer_call_and_return_conditional_losses_23600h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
%__inference_PoolE_layer_call_fn_23585?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
%__inference_PoolE_layer_call_fn_23590[7?4
-?*
(?%
inputs?????????
? " ???????????
 __inference__wrapped_model_20323?H@A:;\]VW??|}vwpqjk??????????????????????????6?3
,?)
'?$
input?????????
? "/?,
*
output ?
output?????????
?
F__inference_concatenate_layer_call_and_return_conditional_losses_24095????
???
???
#? 
inputs/0??????????
#? 
inputs/1??????????
#? 
inputs/2??????????
#? 
inputs/3??????????
#? 
inputs/4??????????
#? 
inputs/5??????????
#? 
inputs/6??????????
#? 
inputs/7??????????
#? 
inputs/8??????????
? "&?#
?
0??????????$
? ?
+__inference_concatenate_layer_call_fn_24081????
???
???
#? 
inputs/0??????????
#? 
inputs/1??????????
#? 
inputs/2??????????
#? 
inputs/3??????????
#? 
inputs/4??????????
#? 
inputs/5??????????
#? 
inputs/6??????????
#? 
inputs/7??????????
#? 
inputs/8??????????
? "???????????$?
D__inference_dropout_1_layer_call_and_return_conditional_losses_23308l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
D__inference_dropout_1_layer_call_and_return_conditional_losses_23320l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
)__inference_dropout_1_layer_call_fn_23298_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
)__inference_dropout_1_layer_call_fn_23303_;?8
1?.
(?%
inputs?????????
p
? " ???????????
D__inference_dropout_2_layer_call_and_return_conditional_losses_23723l;?8
1?.
(?%
inputs?????????


p 
? "-?*
#? 
0?????????


? ?
D__inference_dropout_2_layer_call_and_return_conditional_losses_23735l;?8
1?.
(?%
inputs?????????


p
? "-?*
#? 
0?????????


? ?
)__inference_dropout_2_layer_call_fn_23713_;?8
1?.
(?%
inputs?????????


p 
? " ??????????

?
)__inference_dropout_2_layer_call_fn_23718_;?8
1?.
(?%
inputs?????????


p
? " ??????????

?
D__inference_dropout_3_layer_call_and_return_conditional_losses_23750l;?8
1?.
(?%
inputs?????????		
p 
? "-?*
#? 
0?????????		
? ?
D__inference_dropout_3_layer_call_and_return_conditional_losses_23762l;?8
1?.
(?%
inputs?????????		
p
? "-?*
#? 
0?????????		
? ?
)__inference_dropout_3_layer_call_fn_23740_;?8
1?.
(?%
inputs?????????		
p 
? " ??????????		?
)__inference_dropout_3_layer_call_fn_23745_;?8
1?.
(?%
inputs?????????		
p
? " ??????????		?
D__inference_dropout_4_layer_call_and_return_conditional_losses_23777l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
D__inference_dropout_4_layer_call_and_return_conditional_losses_23789l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
)__inference_dropout_4_layer_call_fn_23767_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
)__inference_dropout_4_layer_call_fn_23772_;?8
1?.
(?%
inputs?????????
p
? " ???????????
D__inference_dropout_5_layer_call_and_return_conditional_losses_23615l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
D__inference_dropout_5_layer_call_and_return_conditional_losses_23627l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
)__inference_dropout_5_layer_call_fn_23605_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
)__inference_dropout_5_layer_call_fn_23610_;?8
1?.
(?%
inputs?????????
p
? " ???????????
D__inference_dropout_6_layer_call_and_return_conditional_losses_23642l;?8
1?.
(?%
inputs?????????(
p 
? "-?*
#? 
0?????????(
? ?
D__inference_dropout_6_layer_call_and_return_conditional_losses_23654l;?8
1?.
(?%
inputs?????????(
p
? "-?*
#? 
0?????????(
? ?
)__inference_dropout_6_layer_call_fn_23632_;?8
1?.
(?%
inputs?????????(
p 
? " ??????????(?
)__inference_dropout_6_layer_call_fn_23637_;?8
1?.
(?%
inputs?????????(
p
? " ??????????(?
D__inference_dropout_7_layer_call_and_return_conditional_losses_23669l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
D__inference_dropout_7_layer_call_and_return_conditional_losses_23681l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
)__inference_dropout_7_layer_call_fn_23659_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
)__inference_dropout_7_layer_call_fn_23664_;?8
1?.
(?%
inputs?????????
p
? " ???????????
D__inference_dropout_8_layer_call_and_return_conditional_losses_23696l;?8
1?.
(?%
inputs?????????(
p 
? "-?*
#? 
0?????????(
? ?
D__inference_dropout_8_layer_call_and_return_conditional_losses_23708l;?8
1?.
(?%
inputs?????????(
p
? "-?*
#? 
0?????????(
? ?
)__inference_dropout_8_layer_call_fn_23686_;?8
1?.
(?%
inputs?????????(
p 
? " ??????????(?
)__inference_dropout_8_layer_call_fn_23691_;?8
1?.
(?%
inputs?????????(
p
? " ??????????(?
B__inference_dropout_layer_call_and_return_conditional_losses_23281l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_23293l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
'__inference_dropout_layer_call_fn_23271_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
'__inference_dropout_layer_call_fn_23276_;?8
1?.
(?%
inputs?????????
p
? " ???????????
D__inference_flatten_1_layer_call_and_return_conditional_losses_23811a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
)__inference_flatten_1_layer_call_fn_23805T7?4
-?*
(?%
inputs?????????
? "????????????
D__inference_flatten_2_layer_call_and_return_conditional_losses_23822`7?4
-?*
(?%
inputs?????????(
? "%?"
?
0?????????(
? ?
)__inference_flatten_2_layer_call_fn_23816S7?4
-?*
(?%
inputs?????????(
? "??????????(?
D__inference_flatten_3_layer_call_and_return_conditional_losses_23833a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
)__inference_flatten_3_layer_call_fn_23827T7?4
-?*
(?%
inputs?????????
? "????????????
D__inference_flatten_4_layer_call_and_return_conditional_losses_23844a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
)__inference_flatten_4_layer_call_fn_23838T7?4
-?*
(?%
inputs?????????
? "????????????
D__inference_flatten_5_layer_call_and_return_conditional_losses_23855`7?4
-?*
(?%
inputs?????????(
? "%?"
?
0?????????(
? ?
)__inference_flatten_5_layer_call_fn_23849S7?4
-?*
(?%
inputs?????????(
? "??????????(?
D__inference_flatten_6_layer_call_and_return_conditional_losses_23866a7?4
-?*
(?%
inputs?????????


? "&?#
?
0??????????
? ?
)__inference_flatten_6_layer_call_fn_23860T7?4
-?*
(?%
inputs?????????


? "????????????
D__inference_flatten_7_layer_call_and_return_conditional_losses_23877a7?4
-?*
(?%
inputs?????????		
? "&?#
?
0??????????
? ?
)__inference_flatten_7_layer_call_fn_23871T7?4
-?*
(?%
inputs?????????		
? "????????????
D__inference_flatten_8_layer_call_and_return_conditional_losses_23888a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
)__inference_flatten_8_layer_call_fn_23882T7?4
-?*
(?%
inputs?????????
? "????????????
B__inference_flatten_layer_call_and_return_conditional_losses_23800a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? 
'__inference_flatten_layer_call_fn_23794T7?4
-?*
(?%
inputs?????????
? "????????????
@__inference_model_layer_call_and_return_conditional_losses_22302?H@A:;\]VW??|}vwpqjk??????????????????????????>?;
4?1
'?$
input?????????
p 

 
? "%?"
?
0?????????

? ?
@__inference_model_layer_call_and_return_conditional_losses_22444?H@A:;\]VW??|}vwpqjk??????????????????????????>?;
4?1
'?$
input?????????
p

 
? "%?"
?
0?????????

? ?
@__inference_model_layer_call_and_return_conditional_losses_22927?H@A:;\]VW??|}vwpqjk????????????????????????????<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????

? ?
@__inference_model_layer_call_and_return_conditional_losses_23186?H@A:;\]VW??|}vwpqjk????????????????????????????<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????

? ?
%__inference_model_layer_call_fn_21199?H@A:;\]VW??|}vwpqjk??????????????????????????>?;
4?1
'?$
input?????????
p 

 
? "??????????
?
%__inference_model_layer_call_fn_22160?H@A:;\]VW??|}vwpqjk??????????????????????????>?;
4?1
'?$
input?????????
p

 
? "??????????
?
%__inference_model_layer_call_fn_22638?H@A:;\]VW??|}vwpqjk????????????????????????????<
5?2
(?%
inputs?????????
p 

 
? "??????????
?
%__inference_model_layer_call_fn_22731?H@A:;\]VW??|}vwpqjk????????????????????????????<
5?2
(?%
inputs?????????
p

 
? "??????????
?
A__inference_output_layer_call_and_return_conditional_losses_24175_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? |
&__inference_output_layer_call_fn_24164R??0?-
&?#
!?
inputs??????????
? "??????????
?
#__inference_signature_wrapper_22545?H@A:;\]VW??|}vwpqjk????????????????????????????<
? 
5?2
0
input'?$
input?????????"/?,
*
output ?
output?????????
