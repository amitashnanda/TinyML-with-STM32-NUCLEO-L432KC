)
ý
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
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
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.2.02unknown8ä%

prune_low_magnitude_conv1d/maskVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!prune_low_magnitude_conv1d/mask

3prune_low_magnitude_conv1d/mask/Read/ReadVariableOpReadVariableOpprune_low_magnitude_conv1d/mask*"
_output_shapes
:*
dtype0

$prune_low_magnitude_conv1d/thresholdVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$prune_low_magnitude_conv1d/threshold

8prune_low_magnitude_conv1d/threshold/Read/ReadVariableOpReadVariableOp$prune_low_magnitude_conv1d/threshold*
_output_shapes
: *
dtype0
¢
'prune_low_magnitude_conv1d/pruning_stepVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *8
shared_name)'prune_low_magnitude_conv1d/pruning_step

;prune_low_magnitude_conv1d/pruning_step/Read/ReadVariableOpReadVariableOp'prune_low_magnitude_conv1d/pruning_step*
_output_shapes
: *
dtype0	
¢
!prune_low_magnitude_conv1d_1/maskVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!prune_low_magnitude_conv1d_1/mask

5prune_low_magnitude_conv1d_1/mask/Read/ReadVariableOpReadVariableOp!prune_low_magnitude_conv1d_1/mask*"
_output_shapes
:*
dtype0
 
&prune_low_magnitude_conv1d_1/thresholdVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&prune_low_magnitude_conv1d_1/threshold

:prune_low_magnitude_conv1d_1/threshold/Read/ReadVariableOpReadVariableOp&prune_low_magnitude_conv1d_1/threshold*
_output_shapes
: *
dtype0
¦
)prune_low_magnitude_conv1d_1/pruning_stepVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *:
shared_name+)prune_low_magnitude_conv1d_1/pruning_step

=prune_low_magnitude_conv1d_1/pruning_step/Read/ReadVariableOpReadVariableOp)prune_low_magnitude_conv1d_1/pruning_step*
_output_shapes
: *
dtype0	
¢
!prune_low_magnitude_conv1d_2/maskVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!prune_low_magnitude_conv1d_2/mask

5prune_low_magnitude_conv1d_2/mask/Read/ReadVariableOpReadVariableOp!prune_low_magnitude_conv1d_2/mask*"
_output_shapes
:*
dtype0
 
&prune_low_magnitude_conv1d_2/thresholdVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&prune_low_magnitude_conv1d_2/threshold

:prune_low_magnitude_conv1d_2/threshold/Read/ReadVariableOpReadVariableOp&prune_low_magnitude_conv1d_2/threshold*
_output_shapes
: *
dtype0
¦
)prune_low_magnitude_conv1d_2/pruning_stepVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *:
shared_name+)prune_low_magnitude_conv1d_2/pruning_step

=prune_low_magnitude_conv1d_2/pruning_step/Read/ReadVariableOpReadVariableOp)prune_low_magnitude_conv1d_2/pruning_step*
_output_shapes
: *
dtype0	
¤
(prune_low_magnitude_dropout/pruning_stepVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *9
shared_name*(prune_low_magnitude_dropout/pruning_step

<prune_low_magnitude_dropout/pruning_step/Read/ReadVariableOpReadVariableOp(prune_low_magnitude_dropout/pruning_step*
_output_shapes
: *
dtype0	
¤
(prune_low_magnitude_flatten/pruning_stepVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *9
shared_name*(prune_low_magnitude_flatten/pruning_step

<prune_low_magnitude_flatten/pruning_step/Read/ReadVariableOpReadVariableOp(prune_low_magnitude_flatten/pruning_step*
_output_shapes
: *
dtype0	

prune_low_magnitude_dense/maskVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*/
shared_name prune_low_magnitude_dense/mask

2prune_low_magnitude_dense/mask/Read/ReadVariableOpReadVariableOpprune_low_magnitude_dense/mask* 
_output_shapes
:
*
dtype0

#prune_low_magnitude_dense/thresholdVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#prune_low_magnitude_dense/threshold

7prune_low_magnitude_dense/threshold/Read/ReadVariableOpReadVariableOp#prune_low_magnitude_dense/threshold*
_output_shapes
: *
dtype0
 
&prune_low_magnitude_dense/pruning_stepVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *7
shared_name(&prune_low_magnitude_dense/pruning_step

:prune_low_magnitude_dense/pruning_step/Read/ReadVariableOpReadVariableOp&prune_low_magnitude_dense/pruning_step*
_output_shapes
: *
dtype0	
\
iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameiter
U
iter/Read/ReadVariableOpReadVariableOpiter*
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
z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
:*
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
:*
dtype0
~
conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_1/kernel
w
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*"
_output_shapes
:*
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
:*
dtype0
~
conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_2/kernel
w
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*"
_output_shapes
:*
dtype0
r
conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_2/bias
k
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes
:*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
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
Adam/conv1d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d/kernel/m

(Adam/conv1d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/m*"
_output_shapes
:*
dtype0
|
Adam/conv1d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv1d/bias/m
u
&Adam/conv1d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/m*
_output_shapes
:*
dtype0

Adam/conv1d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv1d_1/kernel/m

*Adam/conv1d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/kernel/m*"
_output_shapes
:*
dtype0

Adam/conv1d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_1/bias/m
y
(Adam/conv1d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/bias/m*
_output_shapes
:*
dtype0

Adam/conv1d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv1d_2/kernel/m

*Adam/conv1d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/kernel/m*"
_output_shapes
:*
dtype0

Adam/conv1d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_2/bias/m
y
(Adam/conv1d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/bias/m*
_output_shapes
:*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0

Adam/conv1d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d/kernel/v

(Adam/conv1d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/v*"
_output_shapes
:*
dtype0
|
Adam/conv1d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv1d/bias/v
u
&Adam/conv1d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/v*
_output_shapes
:*
dtype0

Adam/conv1d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv1d_1/kernel/v

*Adam/conv1d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/kernel/v*"
_output_shapes
:*
dtype0

Adam/conv1d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_1/bias/v
y
(Adam/conv1d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/bias/v*
_output_shapes
:*
dtype0

Adam/conv1d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv1d_2/kernel/v

*Adam/conv1d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/kernel/v*"
_output_shapes
:*
dtype0

Adam/conv1d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_2/bias/v
y
(Adam/conv1d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/bias/v*
_output_shapes
:*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
V
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ÂU
value¸UBµU B®U
Û
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
	optimizer
regularization_losses
	trainable_variables

	variables
	keras_api

signatures
°
pruning_vars
	layer
prunable_weights
mask
	threshold
pruning_step
regularization_losses
trainable_variables
	variables
	keras_api
°
pruning_vars
	layer
prunable_weights
mask
	threshold
pruning_step
regularization_losses
trainable_variables
	variables
 	keras_api
°
!pruning_vars
	"layer
#prunable_weights
$mask
%	threshold
&pruning_step
'regularization_losses
(trainable_variables
)	variables
*	keras_api

+pruning_vars
	,layer
-prunable_weights
.pruning_step
/regularization_losses
0trainable_variables
1	variables
2	keras_api

3pruning_vars
	4layer
5prunable_weights
6pruning_step
7regularization_losses
8trainable_variables
9	variables
:	keras_api
°
;pruning_vars
	<layer
=prunable_weights
>mask
?	threshold
@pruning_step
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
à
Eiter

Fbeta_1

Gbeta_2
	Hdecay
Ilearning_rateJmºKm»Lm¼Mm½Nm¾Om¿PmÀQmÁJvÂKvÃLvÄMvÅNvÆOvÇPvÈQvÉ
 
8
J0
K1
L2
M3
N4
O5
P6
Q7
¦
J0
K1
2
3
4
L5
M6
7
8
9
N10
O11
$12
%13
&14
.15
616
P17
Q18
>19
?20
@21
­
regularization_losses
	trainable_variables
Rmetrics
Slayer_regularization_losses
Tnon_trainable_variables

Ulayers
Vlayer_metrics

	variables
 

W0
h

Jkernel
Kbias
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api

J0
ig
VARIABLE_VALUEprune_low_magnitude_conv1d/mask4layer_with_weights-0/mask/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE$prune_low_magnitude_conv1d/threshold9layer_with_weights-0/threshold/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE'prune_low_magnitude_conv1d/pruning_step<layer_with_weights-0/pruning_step/.ATTRIBUTES/VARIABLE_VALUE
 

J0
K1
#
J0
K1
2
3
4
­
regularization_losses
trainable_variables
\layer_regularization_losses
]metrics
^non_trainable_variables

_layers
`layer_metrics
	variables

a0
h

Lkernel
Mbias
bregularization_losses
ctrainable_variables
d	variables
e	keras_api

L0
ki
VARIABLE_VALUE!prune_low_magnitude_conv1d_1/mask4layer_with_weights-1/mask/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE&prune_low_magnitude_conv1d_1/threshold9layer_with_weights-1/threshold/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE)prune_low_magnitude_conv1d_1/pruning_step<layer_with_weights-1/pruning_step/.ATTRIBUTES/VARIABLE_VALUE
 

L0
M1
#
L0
M1
2
3
4
­
regularization_losses
trainable_variables
flayer_regularization_losses
gmetrics
hnon_trainable_variables

ilayers
jlayer_metrics
	variables

k0
h

Nkernel
Obias
lregularization_losses
mtrainable_variables
n	variables
o	keras_api

N0
ki
VARIABLE_VALUE!prune_low_magnitude_conv1d_2/mask4layer_with_weights-2/mask/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE&prune_low_magnitude_conv1d_2/threshold9layer_with_weights-2/threshold/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE)prune_low_magnitude_conv1d_2/pruning_step<layer_with_weights-2/pruning_step/.ATTRIBUTES/VARIABLE_VALUE
 

N0
O1
#
N0
O1
$2
%3
&4
­
'regularization_losses
(trainable_variables
player_regularization_losses
qmetrics
rnon_trainable_variables

slayers
tlayer_metrics
)	variables
 
R
uregularization_losses
vtrainable_variables
w	variables
x	keras_api
 
zx
VARIABLE_VALUE(prune_low_magnitude_dropout/pruning_step<layer_with_weights-3/pruning_step/.ATTRIBUTES/VARIABLE_VALUE
 
 

.0
­
/regularization_losses
0trainable_variables
ylayer_regularization_losses
zmetrics
{non_trainable_variables

|layers
}layer_metrics
1	variables
 
T
~regularization_losses
trainable_variables
	variables
	keras_api
 
zx
VARIABLE_VALUE(prune_low_magnitude_flatten/pruning_step<layer_with_weights-4/pruning_step/.ATTRIBUTES/VARIABLE_VALUE
 
 

60
²
7regularization_losses
8trainable_variables
 layer_regularization_losses
metrics
non_trainable_variables
layers
layer_metrics
9	variables

0
l

Pkernel
Qbias
regularization_losses
trainable_variables
	variables
	keras_api

P0
hf
VARIABLE_VALUEprune_low_magnitude_dense/mask4layer_with_weights-5/mask/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE#prune_low_magnitude_dense/threshold9layer_with_weights-5/threshold/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE&prune_low_magnitude_dense/pruning_step<layer_with_weights-5/pruning_step/.ATTRIBUTES/VARIABLE_VALUE
 

P0
Q1
#
P0
Q1
>2
?3
@4
²
Aregularization_losses
Btrainable_variables
 layer_regularization_losses
metrics
non_trainable_variables
layers
layer_metrics
C	variables
CA
VARIABLE_VALUEiter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv1d/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEconv1d/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv1d_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv1d_1/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv1d_2/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv1d_2/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUE
dense/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
f
0
1
2
3
4
5
$6
%7
&8
.9
610
>11
?12
@13
*
0
1
2
3
4
5
 

J0
1
2
 

J0
K1

J0
K1
²
Xregularization_losses
Ytrainable_variables
 layer_regularization_losses
metrics
non_trainable_variables
layers
layer_metrics
Z	variables
 
 

0
1
2

0
 

L0
1
2
 

L0
M1

L0
M1
²
bregularization_losses
ctrainable_variables
 layer_regularization_losses
metrics
non_trainable_variables
layers
layer_metrics
d	variables
 
 

0
1
2

0
 

N0
$1
%2
 

N0
O1

N0
O1
²
lregularization_losses
mtrainable_variables
 layer_regularization_losses
metrics
non_trainable_variables
 layers
¡layer_metrics
n	variables
 
 

$0
%1
&2

"0
 
 
 
 
²
uregularization_losses
vtrainable_variables
 ¢layer_regularization_losses
£metrics
¤non_trainable_variables
¥layers
¦layer_metrics
w	variables
 
 

.0

,0
 
 
 
 
³
~regularization_losses
trainable_variables
 §layer_regularization_losses
¨metrics
©non_trainable_variables
ªlayers
«layer_metrics
	variables
 
 

60

40
 

P0
>1
?2
 

P0
Q1

P0
Q1
µ
regularization_losses
trainable_variables
 ¬layer_regularization_losses
­metrics
®non_trainable_variables
¯layers
°layer_metrics
	variables
 
 

>0
?1
@2

<0
 
8

±total

²count
³	variables
´	keras_api
I

µtotal

¶count
·
_fn_kwargs
¸	variables
¹	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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
±0
²1

³	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

µ0
¶1

¸	variables
vt
VARIABLE_VALUEAdam/conv1d/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/conv1d/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv1d_1/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv1d_1/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv1d_2/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv1d_2/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dense/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv1d/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/conv1d/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv1d_1/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv1d_1/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv1d_2/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv1d_2/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dense/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_conv1d_inputPlaceholder*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	*
dtype0*!
shape:ÿÿÿÿÿÿÿÿÿâ	
¯
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv1d_inputconv1d/kernelprune_low_magnitude_conv1d/maskconv1d/biasconv1d_1/kernel!prune_low_magnitude_conv1d_1/maskconv1d_1/biasconv1d_2/kernel!prune_low_magnitude_conv1d_2/maskconv1d_2/biasdense/kernelprune_low_magnitude_dense/mask
dense/bias*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

	*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*-
f(R&
$__inference_signature_wrapper_317404
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¬
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename3prune_low_magnitude_conv1d/mask/Read/ReadVariableOp8prune_low_magnitude_conv1d/threshold/Read/ReadVariableOp;prune_low_magnitude_conv1d/pruning_step/Read/ReadVariableOp5prune_low_magnitude_conv1d_1/mask/Read/ReadVariableOp:prune_low_magnitude_conv1d_1/threshold/Read/ReadVariableOp=prune_low_magnitude_conv1d_1/pruning_step/Read/ReadVariableOp5prune_low_magnitude_conv1d_2/mask/Read/ReadVariableOp:prune_low_magnitude_conv1d_2/threshold/Read/ReadVariableOp=prune_low_magnitude_conv1d_2/pruning_step/Read/ReadVariableOp<prune_low_magnitude_dropout/pruning_step/Read/ReadVariableOp<prune_low_magnitude_flatten/pruning_step/Read/ReadVariableOp2prune_low_magnitude_dense/mask/Read/ReadVariableOp7prune_low_magnitude_dense/threshold/Read/ReadVariableOp:prune_low_magnitude_dense/pruning_step/Read/ReadVariableOpiter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp#conv1d_2/kernel/Read/ReadVariableOp!conv1d_2/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/conv1d/kernel/m/Read/ReadVariableOp&Adam/conv1d/bias/m/Read/ReadVariableOp*Adam/conv1d_1/kernel/m/Read/ReadVariableOp(Adam/conv1d_1/bias/m/Read/ReadVariableOp*Adam/conv1d_2/kernel/m/Read/ReadVariableOp(Adam/conv1d_2/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp(Adam/conv1d/kernel/v/Read/ReadVariableOp&Adam/conv1d/bias/v/Read/ReadVariableOp*Adam/conv1d_1/kernel/v/Read/ReadVariableOp(Adam/conv1d_1/bias/v/Read/ReadVariableOp*Adam/conv1d_2/kernel/v/Read/ReadVariableOp(Adam/conv1d_2/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst*<
Tin5
321							*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*(
f#R!
__inference__traced_save_319624
û

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameprune_low_magnitude_conv1d/mask$prune_low_magnitude_conv1d/threshold'prune_low_magnitude_conv1d/pruning_step!prune_low_magnitude_conv1d_1/mask&prune_low_magnitude_conv1d_1/threshold)prune_low_magnitude_conv1d_1/pruning_step!prune_low_magnitude_conv1d_2/mask&prune_low_magnitude_conv1d_2/threshold)prune_low_magnitude_conv1d_2/pruning_step(prune_low_magnitude_dropout/pruning_step(prune_low_magnitude_flatten/pruning_stepprune_low_magnitude_dense/mask#prune_low_magnitude_dense/threshold&prune_low_magnitude_dense/pruning_stepiterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasconv1d_2/kernelconv1d_2/biasdense/kernel
dense/biastotalcounttotal_1count_1Adam/conv1d/kernel/mAdam/conv1d/bias/mAdam/conv1d_1/kernel/mAdam/conv1d_1/bias/mAdam/conv1d_2/kernel/mAdam/conv1d_2/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/conv1d/kernel/vAdam/conv1d/bias/vAdam/conv1d_1/kernel/vAdam/conv1d_1/bias/vAdam/conv1d_2/kernel/vAdam/conv1d_2/bias/vAdam/dense/kernel/vAdam/dense/bias/v*;
Tin4
220*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*+
f&R$
"__inference__traced_restore_319777¨$
 
¼
+prune_low_magnitude_dense_cond_false_318150
placeholder
placeholder_1
placeholder_2
placeholder_33
/identity_prune_low_magnitude_dense_logicaland_1


identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOpy
IdentityIdentity/identity_prune_low_magnitude_dense_logicaland_1^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
::::: : 

_output_shapes
: :
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
: 
ûS
Ð
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_319232

inputs
readvariableop_resource
identity¢AssignVariableOp¢'assert_greater_equal/Assert/AssertGuardp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOpP
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
add/y\
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: 2
add
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOpA
updateNoOp^AssignVariableOp*
_output_shapes
 2
update­
#assert_greater_equal/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
: *
dtype0	2%
#assert_greater_equal/ReadVariableOpr
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
assert_greater_equal/yÅ
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2#
!assert_greater_equal/GreaterEqual{
assert_greater_equal/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
assert_greater_equal/Const
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/Const:output:0*
_output_shapes
: 2
assert_greater_equal/All
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2#
!assert_greater_equal/Assert/Const¶
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2%
#assert_greater_equal/Assert/Const_1·
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2%
#assert_greater_equal/Assert/Const_2ª
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2%
#assert_greater_equal/Assert/Const_3
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_319161*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_3191602)
'assert_greater_equal/Assert/AssertGuardÃ
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identityú
0polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	22
0polynomial_decay_pruning_schedule/ReadVariableOpÇ
'polynomial_decay_pruning_schedule/sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'polynomial_decay_pruning_schedule/sub/yâ
%polynomial_decay_pruning_schedule/subSub8polynomial_decay_pruning_schedule/ReadVariableOp:value:00polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/sub³
&polynomial_decay_pruning_schedule/CastCast)polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2(
&polynomial_decay_pruning_schedule/CastÒ
+polynomial_decay_pruning_schedule/truediv/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 * ¨G2-
+polynomial_decay_pruning_schedule/truediv/yä
)polynomial_decay_pruning_schedule/truedivRealDiv*polynomial_decay_pruning_schedule/Cast:y:04polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/truedivÒ
+polynomial_decay_pruning_schedule/Maximum/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *    2-
+polynomial_decay_pruning_schedule/Maximum/xç
)polynomial_decay_pruning_schedule/MaximumMaximum4polynomial_decay_pruning_schedule/Maximum/x:output:0-polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/MaximumÒ
+polynomial_decay_pruning_schedule/Minimum/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+polynomial_decay_pruning_schedule/Minimum/xç
)polynomial_decay_pruning_schedule/MinimumMinimum4polynomial_decay_pruning_schedule/Minimum/x:output:0-polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/MinimumÎ
)polynomial_decay_pruning_schedule/sub_1/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)polynomial_decay_pruning_schedule/sub_1/xÝ
'polynomial_decay_pruning_schedule/sub_1Sub2polynomial_decay_pruning_schedule/sub_1/x:output:0-polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2)
'polynomial_decay_pruning_schedule/sub_1Ê
'polynomial_decay_pruning_schedule/Pow/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  @@2)
'polynomial_decay_pruning_schedule/Pow/yÕ
%polynomial_decay_pruning_schedule/PowPow+polynomial_decay_pruning_schedule/sub_1:z:00polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/PowÊ
'polynomial_decay_pruning_schedule/Mul/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *¾2)
'polynomial_decay_pruning_schedule/Mul/xÓ
%polynomial_decay_pruning_schedule/MulMul0polynomial_decay_pruning_schedule/Mul/x:output:0)polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/MulÔ
,polynomial_decay_pruning_schedule/sparsity/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2.
,polynomial_decay_pruning_schedule/sparsity/yâ
*polynomial_decay_pruning_schedule/sparsityAdd)polynomial_decay_pruning_schedule/Mul:z:05polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2,
*polynomial_decay_pruning_schedule/sparsityÐ
GreaterEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOp
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqualÊ
LessEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqual
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
Less/x
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0	*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd¾
Sub/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOp
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
Sub

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorMod
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1¦
condStatelessIfLogicalAnd_1:z:0LogicalAnd_1:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *$
else_branchR
cond_false_319218*
output_shapes
: *#
then_branchR
cond_true_3192172
condZ
cond/IdentityIdentitycond:output:0*
T0
*
_output_shapes
: 2
cond/Identityu
update_1NoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 2

update_16

group_depsNoOp*
_output_shapes
 2

group_deps_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿM  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshape£
IdentityIdentityReshape:output:0^AssignVariableOp(^assert_greater_equal/Assert/AssertGuard*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÙ	:2$
AssignVariableOpAssignVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	
 
_user_specified_nameinputs:

_output_shapes
: 
ù
¡
=__inference_prune_low_magnitude_conv1d_1_layer_call_fn_318817

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	*$
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*a
f\RZ
X__inference_prune_low_magnitude_conv1d_1_layer_call_and_return_conditional_losses_3164492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿà	:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ò

cond_false_317018
placeholder
placeholder_1
placeholder_2
placeholder_3
identity_logicaland_1


identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOp_
IdentityIdentityidentity_logicaland_1^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
::::: : 

_output_shapes
: :
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
: 

¹
D__inference_conv1d_1_layer_call_and_return_conditional_losses_315989

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityp
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1À
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
í
Ä
Mprune_low_magnitude_dense_assert_greater_equal_Assert_AssertGuard_true_318092?
;identity_prune_low_magnitude_dense_assert_greater_equal_all

placeholder	
placeholder_1	

identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOp
IdentityIdentity;identity_prune_low_magnitude_dense_assert_greater_equal_all^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


3assert_greater_equal_Assert_AssertGuard_true_316046%
!identity_assert_greater_equal_all

placeholder	
placeholder_1	

identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOpk
IdentityIdentity!identity_assert_greater_equal_all^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ã
Ê
4assert_greater_equal_Assert_AssertGuard_false_316276#
assert_assert_greater_equal_all
.
*assert_assert_greater_equal_readvariableop	!
assert_assert_greater_equal_y	

identity_1
¢Assertî
Assert/data_0Const*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2
Assert/data_0
Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2
Assert/data_1
Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2
Assert/data_2~
Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2
Assert/data_4
AssertAssertassert_assert_greater_equal_allAssert/data_0:output:0Assert/data_1:output:0Assert/data_2:output:0*assert_assert_greater_equal_readvariableopAssert/data_4:output:0assert_assert_greater_equal_y*
T

2		*
_output_shapes
 2
Assertk
IdentityIdentityassert_assert_greater_equal_all^Assert*
T0
*
_output_shapes
: 2

Identitya

Identity_1IdentityIdentity:output:0^Assert*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : 2
AssertAssert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ò

cond_false_316333
placeholder
placeholder_1
placeholder_2
placeholder_3
identity_logicaland_1


identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOp_
IdentityIdentityidentity_logicaland_1^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
::::: : 

_output_shapes
: :
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
: 
ó
È
Oprune_low_magnitude_flatten_assert_greater_equal_Assert_AssertGuard_true_318008A
=identity_prune_low_magnitude_flatten_assert_greater_equal_all

placeholder	
placeholder_1	

identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOp
IdentityIdentity=identity_prune_low_magnitude_flatten_assert_greater_equal_all^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ã
Ê
4assert_greater_equal_Assert_AssertGuard_false_319267#
assert_assert_greater_equal_all
.
*assert_assert_greater_equal_readvariableop	!
assert_assert_greater_equal_y	

identity_1
¢Assertî
Assert/data_0Const*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2
Assert/data_0
Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2
Assert/data_1
Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2
Assert/data_2~
Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2
Assert/data_4
AssertAssertassert_assert_greater_equal_allAssert/data_0:output:0Assert/data_1:output:0Assert/data_2:output:0*assert_assert_greater_equal_readvariableopAssert/data_4:output:0assert_assert_greater_equal_y*
T

2		*
_output_shapes
 2
Assertk
IdentityIdentityassert_assert_greater_equal_all^Assert*
T0
*
_output_shapes
: 2

Identitya

Identity_1IdentityIdentity:output:0^Assert*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : 2
AssertAssert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

¾
Qprune_low_magnitude_conv1d_2_assert_greater_equal_Assert_AssertGuard_false_317753@
<assert_prune_low_magnitude_conv1d_2_assert_greater_equal_all
K
Gassert_prune_low_magnitude_conv1d_2_assert_greater_equal_readvariableop	>
:assert_prune_low_magnitude_conv1d_2_assert_greater_equal_y	

identity_1
¢Assertî
Assert/data_0Const*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2
Assert/data_0
Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2
Assert/data_1¨
Assert/data_2Const*
_output_shapes
: *
dtype0*Z
valueQBO BIx (prune_low_magnitude_conv1d_2/assert_greater_equal/ReadVariableOp:0) = 2
Assert/data_2
Assert/data_4Const*
_output_shapes
: *
dtype0*M
valueDBB B<y (prune_low_magnitude_conv1d_2/assert_greater_equal/y:0) = 2
Assert/data_4ä
AssertAssert<assert_prune_low_magnitude_conv1d_2_assert_greater_equal_allAssert/data_0:output:0Assert/data_1:output:0Assert/data_2:output:0Gassert_prune_low_magnitude_conv1d_2_assert_greater_equal_readvariableopAssert/data_4:output:0:assert_prune_low_magnitude_conv1d_2_assert_greater_equal_y*
T

2		*
_output_shapes
 2
Assert
IdentityIdentity<assert_prune_low_magnitude_conv1d_2_assert_greater_equal_all^Assert*
T0
*
_output_shapes
: 2

Identitya

Identity_1IdentityIdentity:output:0^Assert*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : 2
AssertAssert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ùf
Æ
V__inference_prune_low_magnitude_conv1d_layer_call_and_return_conditional_losses_316200

inputs
readvariableop_resource
cond_input_1
cond_input_2
cond_input_3#
biasadd_readvariableop_resource
identity¢AssignVariableOp¢AssignVariableOp_1¢'assert_greater_equal/Assert/AssertGuard¢condp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOpP
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
add/y\
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: 2
add
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOpA
updateNoOp^AssignVariableOp*
_output_shapes
 2
update­
#assert_greater_equal/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
: *
dtype0	2%
#assert_greater_equal/ReadVariableOpr
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
assert_greater_equal/yÅ
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2#
!assert_greater_equal/GreaterEqual{
assert_greater_equal/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
assert_greater_equal/Const
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/Const:output:0*
_output_shapes
: 2
assert_greater_equal/All
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2#
!assert_greater_equal/Assert/Const¶
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2%
#assert_greater_equal/Assert/Const_1·
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2%
#assert_greater_equal/Assert/Const_2ª
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2%
#assert_greater_equal/Assert/Const_3
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_316047*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_3160462)
'assert_greater_equal/Assert/AssertGuardÃ
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identityú
0polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	22
0polynomial_decay_pruning_schedule/ReadVariableOpÇ
'polynomial_decay_pruning_schedule/sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'polynomial_decay_pruning_schedule/sub/yâ
%polynomial_decay_pruning_schedule/subSub8polynomial_decay_pruning_schedule/ReadVariableOp:value:00polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/sub³
&polynomial_decay_pruning_schedule/CastCast)polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2(
&polynomial_decay_pruning_schedule/CastÒ
+polynomial_decay_pruning_schedule/truediv/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 * ¨G2-
+polynomial_decay_pruning_schedule/truediv/yä
)polynomial_decay_pruning_schedule/truedivRealDiv*polynomial_decay_pruning_schedule/Cast:y:04polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/truedivÒ
+polynomial_decay_pruning_schedule/Maximum/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *    2-
+polynomial_decay_pruning_schedule/Maximum/xç
)polynomial_decay_pruning_schedule/MaximumMaximum4polynomial_decay_pruning_schedule/Maximum/x:output:0-polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/MaximumÒ
+polynomial_decay_pruning_schedule/Minimum/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+polynomial_decay_pruning_schedule/Minimum/xç
)polynomial_decay_pruning_schedule/MinimumMinimum4polynomial_decay_pruning_schedule/Minimum/x:output:0-polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/MinimumÎ
)polynomial_decay_pruning_schedule/sub_1/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)polynomial_decay_pruning_schedule/sub_1/xÝ
'polynomial_decay_pruning_schedule/sub_1Sub2polynomial_decay_pruning_schedule/sub_1/x:output:0-polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2)
'polynomial_decay_pruning_schedule/sub_1Ê
'polynomial_decay_pruning_schedule/Pow/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  @@2)
'polynomial_decay_pruning_schedule/Pow/yÕ
%polynomial_decay_pruning_schedule/PowPow+polynomial_decay_pruning_schedule/sub_1:z:00polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/PowÊ
'polynomial_decay_pruning_schedule/Mul/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *¾2)
'polynomial_decay_pruning_schedule/Mul/xÓ
%polynomial_decay_pruning_schedule/MulMul0polynomial_decay_pruning_schedule/Mul/x:output:0)polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/MulÔ
,polynomial_decay_pruning_schedule/sparsity/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2.
,polynomial_decay_pruning_schedule/sparsity/yâ
*polynomial_decay_pruning_schedule/sparsityAdd)polynomial_decay_pruning_schedule/Mul:z:05polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2,
*polynomial_decay_pruning_schedule/sparsityÐ
GreaterEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOp
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqualÊ
LessEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqual
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
Less/x
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0	*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd¾
Sub/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOp
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
Sub

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorMod
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1û
condIfLogicalAnd_1:z:0readvariableop_resourcecond_input_1cond_input_2cond_input_3LogicalAnd_1:z:0^AssignVariableOp*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*$
else_branchR
cond_false_316104*
output_shapes
: *#
then_branchR
cond_true_3161032
condZ
cond/IdentityIdentitycond:output:0*
T0
*
_output_shapes
: 2
cond/Identityu
update_1NoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 2

update_1y
Mul/ReadVariableOpReadVariableOpcond_input_1*"
_output_shapes
:*
dtype02
Mul/ReadVariableOp
Mul/ReadVariableOp_1ReadVariableOpcond_input_2^cond*"
_output_shapes
:*
dtype02
Mul/ReadVariableOp_1x
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*"
_output_shapes
:2
Mul
AssignVariableOp_1AssignVariableOpcond_input_1Mul:z:0^Mul/ReadVariableOp^cond*
_output_shapes
 *
dtype02
AssignVariableOp_1y

group_depsNoOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_depsu
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2
group_deps_1p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	2
conv1d/ExpandDims®
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOpcond_input_1^AssignVariableOp_1*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1¸
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	*
squeeze_dims
2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	2
ReluÄ
IdentityIdentityRelu:activations:0^AssignVariableOp^AssignVariableOp_1(^assert_greater_equal/Assert/AssertGuard^cond*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿâ	:::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2
condcond:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	
 
_user_specified_nameinputs:
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
: 
éK
È
*prune_low_magnitude_dense_cond_true_318149=
9polynomial_decay_pruning_schedule_readvariableop_resource+
'pruning_ops_abs_readvariableop_resource
assignvariableop_resource
assignvariableop_1_resource3
/identity_prune_low_magnitude_dense_logicaland_1


identity_1
¢AssignVariableOp¢AssignVariableOp_1Ö
0polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	22
0polynomial_decay_pruning_schedule/ReadVariableOp
'polynomial_decay_pruning_schedule/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'polynomial_decay_pruning_schedule/sub/yâ
%polynomial_decay_pruning_schedule/subSub8polynomial_decay_pruning_schedule/ReadVariableOp:value:00polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/sub³
&polynomial_decay_pruning_schedule/CastCast)polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2(
&polynomial_decay_pruning_schedule/Cast
+polynomial_decay_pruning_schedule/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ¨G2-
+polynomial_decay_pruning_schedule/truediv/yä
)polynomial_decay_pruning_schedule/truedivRealDiv*polynomial_decay_pruning_schedule/Cast:y:04polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/truediv
+polynomial_decay_pruning_schedule/Maximum/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+polynomial_decay_pruning_schedule/Maximum/xç
)polynomial_decay_pruning_schedule/MaximumMaximum4polynomial_decay_pruning_schedule/Maximum/x:output:0-polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/Maximum
+polynomial_decay_pruning_schedule/Minimum/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+polynomial_decay_pruning_schedule/Minimum/xç
)polynomial_decay_pruning_schedule/MinimumMinimum4polynomial_decay_pruning_schedule/Minimum/x:output:0-polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/Minimum
)polynomial_decay_pruning_schedule/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)polynomial_decay_pruning_schedule/sub_1/xÝ
'polynomial_decay_pruning_schedule/sub_1Sub2polynomial_decay_pruning_schedule/sub_1/x:output:0-polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2)
'polynomial_decay_pruning_schedule/sub_1
'polynomial_decay_pruning_schedule/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2)
'polynomial_decay_pruning_schedule/Pow/yÕ
%polynomial_decay_pruning_schedule/PowPow+polynomial_decay_pruning_schedule/sub_1:z:00polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/Pow
'polynomial_decay_pruning_schedule/Mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¾2)
'polynomial_decay_pruning_schedule/Mul/xÓ
%polynomial_decay_pruning_schedule/MulMul0polynomial_decay_pruning_schedule/Mul/x:output:0)polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/Mul¡
,polynomial_decay_pruning_schedule/sparsity/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2.
,polynomial_decay_pruning_schedule/sparsity/yâ
*polynomial_decay_pruning_schedule/sparsityAdd)polynomial_decay_pruning_schedule/Mul:z:05polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2,
*polynomial_decay_pruning_schedule/sparsity¬
GreaterEqual/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOpb
GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqual¦
LessEqual/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp^
LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqualT
Less/xConst*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
Less/xR
Less/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0	*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd
Sub/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOpP
Sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
SubZ

FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorModT
Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1ª
pruning_ops/Abs/ReadVariableOpReadVariableOp'pruning_ops_abs_readvariableop_resource* 
_output_shapes
:
*
dtype02 
pruning_ops/Abs/ReadVariableOp|
pruning_ops/AbsAbs&pruning_ops/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2
pruning_ops/Absh
pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
valueB	 :2
pruning_ops/Sizew
pruning_ops/CastCastpruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
pruning_ops/Castk
pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
pruning_ops/sub/x
pruning_ops/subSubpruning_ops/sub/x:output:0.polynomial_decay_pruning_schedule/sparsity:z:0*
T0*
_output_shapes
: 2
pruning_ops/subu
pruning_ops/mulMulpruning_ops/Cast:y:0pruning_ops/sub:z:0*
T0*
_output_shapes
: 2
pruning_ops/mule
pruning_ops/RoundRoundpruning_ops/mul:z:0*
T0*
_output_shapes
: 2
pruning_ops/Rounds
pruning_ops/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
pruning_ops/Maximum/y
pruning_ops/MaximumMaximumpruning_ops/Round:y:0pruning_ops/Maximum/y:output:0*
T0*
_output_shapes
: 2
pruning_ops/Maximumy
pruning_ops/Cast_1Castpruning_ops/Maximum:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
pruning_ops/Cast_1
pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
pruning_ops/Reshape/shape
pruning_ops/ReshapeReshapepruning_ops/Abs:y:0"pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes

:2
pruning_ops/Reshapel
pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
valueB	 :2
pruning_ops/Size_1
pruning_ops/TopKV2TopKV2pruning_ops/Reshape:output:0pruning_ops/Size_1:output:0*
T0*$
_output_shapes
::2
pruning_ops/TopKV2l
pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
pruning_ops/sub_1/y
pruning_ops/sub_1Subpruning_ops/Cast_1:y:0pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: 2
pruning_ops/sub_1x
pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
pruning_ops/GatherV2/axisÔ
pruning_ops/GatherV2GatherV2pruning_ops/TopKV2:values:0pruning_ops/sub_1:z:0"pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: 2
pruning_ops/GatherV2£
pruning_ops/GreaterEqualGreaterEqualpruning_ops/Abs:y:0pruning_ops/GatherV2:output:0*
T0* 
_output_shapes
:
2
pruning_ops/GreaterEqual
pruning_ops/Cast_2Castpruning_ops/GreaterEqual:z:0*

DstT0*

SrcT0
* 
_output_shapes
:
2
pruning_ops/Cast_2
AssignVariableOpAssignVariableOpassignvariableop_resourcepruning_ops/Cast_2:y:0*
_output_shapes
 *
dtype02
AssignVariableOp
AssignVariableOp_1AssignVariableOpassignvariableop_1_resourcepruning_ops/GatherV2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1

group_depsNoOp^AssignVariableOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_deps
IdentityIdentity/identity_prune_low_magnitude_dense_logicaland_1^group_deps*
T0
*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^AssignVariableOp^AssignVariableOp_1*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
::::: 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_1: 

_output_shapes
: :
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
: 
ý
º
Pprune_low_magnitude_flatten_assert_greater_equal_Assert_AssertGuard_false_318009?
;assert_prune_low_magnitude_flatten_assert_greater_equal_all
J
Fassert_prune_low_magnitude_flatten_assert_greater_equal_readvariableop	=
9assert_prune_low_magnitude_flatten_assert_greater_equal_y	

identity_1
¢Assertî
Assert/data_0Const*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2
Assert/data_0
Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2
Assert/data_1§
Assert/data_2Const*
_output_shapes
: *
dtype0*Y
valuePBN BHx (prune_low_magnitude_flatten/assert_greater_equal/ReadVariableOp:0) = 2
Assert/data_2
Assert/data_4Const*
_output_shapes
: *
dtype0*L
valueCBA B;y (prune_low_magnitude_flatten/assert_greater_equal/y:0) = 2
Assert/data_4á
AssertAssert;assert_prune_low_magnitude_flatten_assert_greater_equal_allAssert/data_0:output:0Assert/data_1:output:0Assert/data_2:output:0Fassert_prune_low_magnitude_flatten_assert_greater_equal_readvariableopAssert/data_4:output:09assert_prune_low_magnitude_flatten_assert_greater_equal_y*
T

2		*
_output_shapes
 2
Assert
IdentityIdentity;assert_prune_low_magnitude_flatten_assert_greater_equal_all^Assert*
T0
*
_output_shapes
: 2

Identitya

Identity_1IdentityIdentity:output:0^Assert*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : 2
AssertAssert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
å
u
,prune_low_magnitude_flatten_cond_true_3180655
1identity_prune_low_magnitude_flatten_logicaland_1


identity_1
6

group_depsNoOp*
_output_shapes
 2

group_deps
IdentityIdentity1identity_prune_low_magnitude_flatten_logicaland_1^group_deps*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : 

_output_shapes
: 
ã
Ê
4assert_greater_equal_Assert_AssertGuard_false_318402#
assert_assert_greater_equal_all
.
*assert_assert_greater_equal_readvariableop	!
assert_assert_greater_equal_y	

identity_1
¢Assertî
Assert/data_0Const*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2
Assert/data_0
Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2
Assert/data_1
Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2
Assert/data_2~
Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2
Assert/data_4
AssertAssertassert_assert_greater_equal_allAssert/data_0:output:0Assert/data_1:output:0Assert/data_2:output:0*assert_assert_greater_equal_readvariableopAssert/data_4:output:0assert_assert_greater_equal_y*
T

2		*
_output_shapes
 2
Assertk
IdentityIdentityassert_assert_greater_equal_all^Assert*
T0
*
_output_shapes
: 2

Identitya

Identity_1IdentityIdentity:output:0^Assert*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : 2
AssertAssert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
`
Å
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_317109

inputs
readvariableop_resource
cond_input_1
cond_input_2
cond_input_3#
biasadd_readvariableop_resource
identity¢AssignVariableOp¢AssignVariableOp_1¢'assert_greater_equal/Assert/AssertGuard¢condp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOpP
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
add/y\
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: 2
add
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOpA
updateNoOp^AssignVariableOp*
_output_shapes
 2
update­
#assert_greater_equal/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
: *
dtype0	2%
#assert_greater_equal/ReadVariableOpr
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
assert_greater_equal/yÅ
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2#
!assert_greater_equal/GreaterEqual{
assert_greater_equal/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
assert_greater_equal/Const
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/Const:output:0*
_output_shapes
: 2
assert_greater_equal/All
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2#
!assert_greater_equal/Assert/Const¶
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2%
#assert_greater_equal/Assert/Const_1·
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2%
#assert_greater_equal/Assert/Const_2ª
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2%
#assert_greater_equal/Assert/Const_3
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_316961*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_3169602)
'assert_greater_equal/Assert/AssertGuardÃ
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identityú
0polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	22
0polynomial_decay_pruning_schedule/ReadVariableOpÇ
'polynomial_decay_pruning_schedule/sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'polynomial_decay_pruning_schedule/sub/yâ
%polynomial_decay_pruning_schedule/subSub8polynomial_decay_pruning_schedule/ReadVariableOp:value:00polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/sub³
&polynomial_decay_pruning_schedule/CastCast)polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2(
&polynomial_decay_pruning_schedule/CastÒ
+polynomial_decay_pruning_schedule/truediv/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 * ¨G2-
+polynomial_decay_pruning_schedule/truediv/yä
)polynomial_decay_pruning_schedule/truedivRealDiv*polynomial_decay_pruning_schedule/Cast:y:04polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/truedivÒ
+polynomial_decay_pruning_schedule/Maximum/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *    2-
+polynomial_decay_pruning_schedule/Maximum/xç
)polynomial_decay_pruning_schedule/MaximumMaximum4polynomial_decay_pruning_schedule/Maximum/x:output:0-polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/MaximumÒ
+polynomial_decay_pruning_schedule/Minimum/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+polynomial_decay_pruning_schedule/Minimum/xç
)polynomial_decay_pruning_schedule/MinimumMinimum4polynomial_decay_pruning_schedule/Minimum/x:output:0-polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/MinimumÎ
)polynomial_decay_pruning_schedule/sub_1/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)polynomial_decay_pruning_schedule/sub_1/xÝ
'polynomial_decay_pruning_schedule/sub_1Sub2polynomial_decay_pruning_schedule/sub_1/x:output:0-polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2)
'polynomial_decay_pruning_schedule/sub_1Ê
'polynomial_decay_pruning_schedule/Pow/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  @@2)
'polynomial_decay_pruning_schedule/Pow/yÕ
%polynomial_decay_pruning_schedule/PowPow+polynomial_decay_pruning_schedule/sub_1:z:00polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/PowÊ
'polynomial_decay_pruning_schedule/Mul/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *¾2)
'polynomial_decay_pruning_schedule/Mul/xÓ
%polynomial_decay_pruning_schedule/MulMul0polynomial_decay_pruning_schedule/Mul/x:output:0)polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/MulÔ
,polynomial_decay_pruning_schedule/sparsity/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2.
,polynomial_decay_pruning_schedule/sparsity/yâ
*polynomial_decay_pruning_schedule/sparsityAdd)polynomial_decay_pruning_schedule/Mul:z:05polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2,
*polynomial_decay_pruning_schedule/sparsityÐ
GreaterEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOp
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqualÊ
LessEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqual
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
Less/x
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0	*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd¾
Sub/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOp
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
Sub

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorMod
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1û
condIfLogicalAnd_1:z:0readvariableop_resourcecond_input_1cond_input_2cond_input_3LogicalAnd_1:z:0^AssignVariableOp*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*$
else_branchR
cond_false_317018*
output_shapes
: *#
then_branchR
cond_true_3170172
condZ
cond/IdentityIdentitycond:output:0*
T0
*
_output_shapes
: 2
cond/Identityu
update_1NoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 2

update_1w
Mul/ReadVariableOpReadVariableOpcond_input_1* 
_output_shapes
:
*
dtype02
Mul/ReadVariableOp
Mul/ReadVariableOp_1ReadVariableOpcond_input_2^cond* 
_output_shapes
:
*
dtype02
Mul/ReadVariableOp_1v
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0* 
_output_shapes
:
2
Mul
AssignVariableOp_1AssignVariableOpcond_input_1Mul:z:0^Mul/ReadVariableOp^cond*
_output_shapes
 *
dtype02
AssignVariableOp_1y

group_depsNoOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_depsu
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2
group_deps_1
MatMul/ReadVariableOpReadVariableOpcond_input_1^AssignVariableOp_1* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoid¸
IdentityIdentitySigmoid:y:0^AssignVariableOp^AssignVariableOp_1(^assert_greater_equal/Assert/AssertGuard^cond*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ:::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2
condcond:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:
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
: 
ó
s
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_319238

inputs
identity4
	no_updateNoOp*
_output_shapes
 2
	no_update8
no_update_1NoOp*
_output_shapes
 2
no_update_16

group_depsNoOp*
_output_shapes
 2

group_deps_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿM  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÙ	:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	
 
_user_specified_nameinputs
ý
½
+__inference_sequential_layer_call_fn_317302
conv1d_input
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
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallconv1d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs

*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3172552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesr
p:ÿÿÿÿÿÿÿÿÿâ	::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	
&
_user_specified_nameconv1d_input:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ò
¿
=__inference_prune_low_magnitude_conv1d_2_layer_call_fn_319022

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	*#
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*a
f\RZ
X__inference_prune_low_magnitude_conv1d_2_layer_call_and_return_conditional_losses_3166582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÝ	:::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	
 
_user_specified_nameinputs:
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
: 
¯
|
'__inference_conv1d_layer_call_fn_315972

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_3159622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ó
È
Oprune_low_magnitude_dropout_assert_greater_equal_Assert_AssertGuard_true_317918A
=identity_prune_low_magnitude_dropout_assert_greater_equal_all

placeholder	
placeholder_1	

identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOp
IdentityIdentity=identity_prune_low_magnitude_dropout_assert_greater_equal_all^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ó
ó
X__inference_prune_low_magnitude_conv1d_1_layer_call_and_return_conditional_losses_318791

inputs
mul_readvariableop_resource!
mul_readvariableop_1_resource#
biasadd_readvariableop_resource
identity¢AssignVariableOp4
	no_updateNoOp*
_output_shapes
 2
	no_update8
no_update_1NoOp*
_output_shapes
 2
no_update_1
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*"
_output_shapes
:*
dtype02
Mul/ReadVariableOp
Mul/ReadVariableOp_1ReadVariableOpmul_readvariableop_1_resource*"
_output_shapes
:*
dtype02
Mul/ReadVariableOp_1x
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*"
_output_shapes
:2
Mul
AssignVariableOpAssignVariableOpmul_readvariableop_resourceMul:z:0^Mul/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpw

group_depsNoOp^AssignVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_depsu
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2
group_deps_1p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	2
conv1d/ExpandDims»
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1¸
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	*
squeeze_dims
2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	2
Relu~
IdentityIdentityRelu:activations:0^AssignVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿà	:::2$
AssignVariableOpAssignVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ü
¼
:__inference_prune_low_magnitude_dense_layer_call_fn_319445

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*^
fYRW
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_3171092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ:::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:
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
: 
ã
Ê
4assert_greater_equal_Assert_AssertGuard_false_316505#
assert_assert_greater_equal_all
.
*assert_assert_greater_equal_readvariableop	!
assert_assert_greater_equal_y	

identity_1
¢Assertî
Assert/data_0Const*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2
Assert/data_0
Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2
Assert/data_1
Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2
Assert/data_2~
Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2
Assert/data_4
AssertAssertassert_assert_greater_equal_allAssert/data_0:output:0Assert/data_1:output:0Assert/data_2:output:0*assert_assert_greater_equal_readvariableopAssert/data_4:output:0assert_assert_greater_equal_y*
T

2		*
_output_shapes
 2
Assertk
IdentityIdentityassert_assert_greater_equal_all^Assert*
T0
*
_output_shapes
: 2

Identitya

Identity_1IdentityIdentity:output:0^Assert*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : 2
AssertAssert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ã
Ê
4assert_greater_equal_Assert_AssertGuard_false_316734#
assert_assert_greater_equal_all
.
*assert_assert_greater_equal_readvariableop	!
assert_assert_greater_equal_y	

identity_1
¢Assertî
Assert/data_0Const*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2
Assert/data_0
Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2
Assert/data_1
Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2
Assert/data_2~
Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2
Assert/data_4
AssertAssertassert_assert_greater_equal_allAssert/data_0:output:0Assert/data_1:output:0Assert/data_2:output:0*assert_assert_greater_equal_readvariableopAssert/data_4:output:0assert_assert_greater_equal_y*
T

2		*
_output_shapes
 2
Assertk
IdentityIdentityassert_assert_greater_equal_all^Assert*
T0
*
_output_shapes
: 2

Identitya

Identity_1IdentityIdentity:output:0^Assert*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : 2
AssertAssert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ã
Ê
4assert_greater_equal_Assert_AssertGuard_false_318834#
assert_assert_greater_equal_all
.
*assert_assert_greater_equal_readvariableop	!
assert_assert_greater_equal_y	

identity_1
¢Assertî
Assert/data_0Const*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2
Assert/data_0
Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2
Assert/data_1
Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2
Assert/data_2~
Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2
Assert/data_4
AssertAssertassert_assert_greater_equal_allAssert/data_0:output:0Assert/data_1:output:0Assert/data_2:output:0*assert_assert_greater_equal_readvariableopAssert/data_4:output:0assert_assert_greater_equal_y*
T

2		*
_output_shapes
 2
Assertk
IdentityIdentityassert_assert_greater_equal_all^Assert*
T0
*
_output_shapes
: 2

Identitya

Identity_1IdentityIdentity:output:0^Assert*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : 2
AssertAssert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
K

cond_true_318674=
9polynomial_decay_pruning_schedule_readvariableop_resource+
'pruning_ops_abs_readvariableop_resource
assignvariableop_resource
assignvariableop_1_resource
identity_logicaland_1


identity_1
¢AssignVariableOp¢AssignVariableOp_1Ö
0polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	22
0polynomial_decay_pruning_schedule/ReadVariableOp
'polynomial_decay_pruning_schedule/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'polynomial_decay_pruning_schedule/sub/yâ
%polynomial_decay_pruning_schedule/subSub8polynomial_decay_pruning_schedule/ReadVariableOp:value:00polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/sub³
&polynomial_decay_pruning_schedule/CastCast)polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2(
&polynomial_decay_pruning_schedule/Cast
+polynomial_decay_pruning_schedule/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ¨G2-
+polynomial_decay_pruning_schedule/truediv/yä
)polynomial_decay_pruning_schedule/truedivRealDiv*polynomial_decay_pruning_schedule/Cast:y:04polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/truediv
+polynomial_decay_pruning_schedule/Maximum/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+polynomial_decay_pruning_schedule/Maximum/xç
)polynomial_decay_pruning_schedule/MaximumMaximum4polynomial_decay_pruning_schedule/Maximum/x:output:0-polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/Maximum
+polynomial_decay_pruning_schedule/Minimum/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+polynomial_decay_pruning_schedule/Minimum/xç
)polynomial_decay_pruning_schedule/MinimumMinimum4polynomial_decay_pruning_schedule/Minimum/x:output:0-polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/Minimum
)polynomial_decay_pruning_schedule/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)polynomial_decay_pruning_schedule/sub_1/xÝ
'polynomial_decay_pruning_schedule/sub_1Sub2polynomial_decay_pruning_schedule/sub_1/x:output:0-polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2)
'polynomial_decay_pruning_schedule/sub_1
'polynomial_decay_pruning_schedule/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2)
'polynomial_decay_pruning_schedule/Pow/yÕ
%polynomial_decay_pruning_schedule/PowPow+polynomial_decay_pruning_schedule/sub_1:z:00polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/Pow
'polynomial_decay_pruning_schedule/Mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¾2)
'polynomial_decay_pruning_schedule/Mul/xÓ
%polynomial_decay_pruning_schedule/MulMul0polynomial_decay_pruning_schedule/Mul/x:output:0)polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/Mul¡
,polynomial_decay_pruning_schedule/sparsity/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2.
,polynomial_decay_pruning_schedule/sparsity/yâ
*polynomial_decay_pruning_schedule/sparsityAdd)polynomial_decay_pruning_schedule/Mul:z:05polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2,
*polynomial_decay_pruning_schedule/sparsity¬
GreaterEqual/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOpb
GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqual¦
LessEqual/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp^
LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqualT
Less/xConst*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
Less/xR
Less/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0	*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd
Sub/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOpP
Sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
SubZ

FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorModT
Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1¬
pruning_ops/Abs/ReadVariableOpReadVariableOp'pruning_ops_abs_readvariableop_resource*"
_output_shapes
:*
dtype02 
pruning_ops/Abs/ReadVariableOp~
pruning_ops/AbsAbs&pruning_ops/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:2
pruning_ops/Absg
pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value
B :2
pruning_ops/Sizew
pruning_ops/CastCastpruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
pruning_ops/Castk
pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
pruning_ops/sub/x
pruning_ops/subSubpruning_ops/sub/x:output:0.polynomial_decay_pruning_schedule/sparsity:z:0*
T0*
_output_shapes
: 2
pruning_ops/subu
pruning_ops/mulMulpruning_ops/Cast:y:0pruning_ops/sub:z:0*
T0*
_output_shapes
: 2
pruning_ops/mule
pruning_ops/RoundRoundpruning_ops/mul:z:0*
T0*
_output_shapes
: 2
pruning_ops/Rounds
pruning_ops/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
pruning_ops/Maximum/y
pruning_ops/MaximumMaximumpruning_ops/Round:y:0pruning_ops/Maximum/y:output:0*
T0*
_output_shapes
: 2
pruning_ops/Maximumy
pruning_ops/Cast_1Castpruning_ops/Maximum:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
pruning_ops/Cast_1
pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
pruning_ops/Reshape/shape
pruning_ops/ReshapeReshapepruning_ops/Abs:y:0"pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes	
:2
pruning_ops/Reshapek
pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value
B :2
pruning_ops/Size_1
pruning_ops/TopKV2TopKV2pruning_ops/Reshape:output:0pruning_ops/Size_1:output:0*
T0*"
_output_shapes
::2
pruning_ops/TopKV2l
pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
pruning_ops/sub_1/y
pruning_ops/sub_1Subpruning_ops/Cast_1:y:0pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: 2
pruning_ops/sub_1x
pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
pruning_ops/GatherV2/axisÔ
pruning_ops/GatherV2GatherV2pruning_ops/TopKV2:values:0pruning_ops/sub_1:z:0"pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: 2
pruning_ops/GatherV2¥
pruning_ops/GreaterEqualGreaterEqualpruning_ops/Abs:y:0pruning_ops/GatherV2:output:0*
T0*"
_output_shapes
:2
pruning_ops/GreaterEqual
pruning_ops/Cast_2Castpruning_ops/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
:2
pruning_ops/Cast_2
AssignVariableOpAssignVariableOpassignvariableop_resourcepruning_ops/Cast_2:y:0*
_output_shapes
 *
dtype02
AssignVariableOp
AssignVariableOp_1AssignVariableOpassignvariableop_1_resourcepruning_ops/GatherV2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1

group_depsNoOp^AssignVariableOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_depse
IdentityIdentityidentity_logicaland_1^group_deps*
T0
*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^AssignVariableOp^AssignVariableOp_1*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
::::: 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_1: 

_output_shapes
: :
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
: 
K

cond_true_316332=
9polynomial_decay_pruning_schedule_readvariableop_resource+
'pruning_ops_abs_readvariableop_resource
assignvariableop_resource
assignvariableop_1_resource
identity_logicaland_1


identity_1
¢AssignVariableOp¢AssignVariableOp_1Ö
0polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	22
0polynomial_decay_pruning_schedule/ReadVariableOp
'polynomial_decay_pruning_schedule/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'polynomial_decay_pruning_schedule/sub/yâ
%polynomial_decay_pruning_schedule/subSub8polynomial_decay_pruning_schedule/ReadVariableOp:value:00polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/sub³
&polynomial_decay_pruning_schedule/CastCast)polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2(
&polynomial_decay_pruning_schedule/Cast
+polynomial_decay_pruning_schedule/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ¨G2-
+polynomial_decay_pruning_schedule/truediv/yä
)polynomial_decay_pruning_schedule/truedivRealDiv*polynomial_decay_pruning_schedule/Cast:y:04polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/truediv
+polynomial_decay_pruning_schedule/Maximum/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+polynomial_decay_pruning_schedule/Maximum/xç
)polynomial_decay_pruning_schedule/MaximumMaximum4polynomial_decay_pruning_schedule/Maximum/x:output:0-polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/Maximum
+polynomial_decay_pruning_schedule/Minimum/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+polynomial_decay_pruning_schedule/Minimum/xç
)polynomial_decay_pruning_schedule/MinimumMinimum4polynomial_decay_pruning_schedule/Minimum/x:output:0-polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/Minimum
)polynomial_decay_pruning_schedule/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)polynomial_decay_pruning_schedule/sub_1/xÝ
'polynomial_decay_pruning_schedule/sub_1Sub2polynomial_decay_pruning_schedule/sub_1/x:output:0-polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2)
'polynomial_decay_pruning_schedule/sub_1
'polynomial_decay_pruning_schedule/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2)
'polynomial_decay_pruning_schedule/Pow/yÕ
%polynomial_decay_pruning_schedule/PowPow+polynomial_decay_pruning_schedule/sub_1:z:00polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/Pow
'polynomial_decay_pruning_schedule/Mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¾2)
'polynomial_decay_pruning_schedule/Mul/xÓ
%polynomial_decay_pruning_schedule/MulMul0polynomial_decay_pruning_schedule/Mul/x:output:0)polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/Mul¡
,polynomial_decay_pruning_schedule/sparsity/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2.
,polynomial_decay_pruning_schedule/sparsity/yâ
*polynomial_decay_pruning_schedule/sparsityAdd)polynomial_decay_pruning_schedule/Mul:z:05polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2,
*polynomial_decay_pruning_schedule/sparsity¬
GreaterEqual/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOpb
GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqual¦
LessEqual/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp^
LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqualT
Less/xConst*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
Less/xR
Less/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0	*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd
Sub/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOpP
Sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
SubZ

FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorModT
Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1¬
pruning_ops/Abs/ReadVariableOpReadVariableOp'pruning_ops_abs_readvariableop_resource*"
_output_shapes
:*
dtype02 
pruning_ops/Abs/ReadVariableOp~
pruning_ops/AbsAbs&pruning_ops/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:2
pruning_ops/Absg
pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value
B :2
pruning_ops/Sizew
pruning_ops/CastCastpruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
pruning_ops/Castk
pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
pruning_ops/sub/x
pruning_ops/subSubpruning_ops/sub/x:output:0.polynomial_decay_pruning_schedule/sparsity:z:0*
T0*
_output_shapes
: 2
pruning_ops/subu
pruning_ops/mulMulpruning_ops/Cast:y:0pruning_ops/sub:z:0*
T0*
_output_shapes
: 2
pruning_ops/mule
pruning_ops/RoundRoundpruning_ops/mul:z:0*
T0*
_output_shapes
: 2
pruning_ops/Rounds
pruning_ops/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
pruning_ops/Maximum/y
pruning_ops/MaximumMaximumpruning_ops/Round:y:0pruning_ops/Maximum/y:output:0*
T0*
_output_shapes
: 2
pruning_ops/Maximumy
pruning_ops/Cast_1Castpruning_ops/Maximum:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
pruning_ops/Cast_1
pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
pruning_ops/Reshape/shape
pruning_ops/ReshapeReshapepruning_ops/Abs:y:0"pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes	
:2
pruning_ops/Reshapek
pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value
B :2
pruning_ops/Size_1
pruning_ops/TopKV2TopKV2pruning_ops/Reshape:output:0pruning_ops/Size_1:output:0*
T0*"
_output_shapes
::2
pruning_ops/TopKV2l
pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
pruning_ops/sub_1/y
pruning_ops/sub_1Subpruning_ops/Cast_1:y:0pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: 2
pruning_ops/sub_1x
pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
pruning_ops/GatherV2/axisÔ
pruning_ops/GatherV2GatherV2pruning_ops/TopKV2:values:0pruning_ops/sub_1:z:0"pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: 2
pruning_ops/GatherV2¥
pruning_ops/GreaterEqualGreaterEqualpruning_ops/Abs:y:0pruning_ops/GatherV2:output:0*
T0*"
_output_shapes
:2
pruning_ops/GreaterEqual
pruning_ops/Cast_2Castpruning_ops/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
:2
pruning_ops/Cast_2
AssignVariableOpAssignVariableOpassignvariableop_resourcepruning_ops/Cast_2:y:0*
_output_shapes
 *
dtype02
AssignVariableOp
AssignVariableOp_1AssignVariableOpassignvariableop_1_resourcepruning_ops/GatherV2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1

group_depsNoOp^AssignVariableOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_depse
IdentityIdentityidentity_logicaland_1^group_deps*
T0
*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^AssignVariableOp^AssignVariableOp_1*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
::::: 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_1: 

_output_shapes
: :
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
: 
K

cond_true_318458=
9polynomial_decay_pruning_schedule_readvariableop_resource+
'pruning_ops_abs_readvariableop_resource
assignvariableop_resource
assignvariableop_1_resource
identity_logicaland_1


identity_1
¢AssignVariableOp¢AssignVariableOp_1Ö
0polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	22
0polynomial_decay_pruning_schedule/ReadVariableOp
'polynomial_decay_pruning_schedule/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'polynomial_decay_pruning_schedule/sub/yâ
%polynomial_decay_pruning_schedule/subSub8polynomial_decay_pruning_schedule/ReadVariableOp:value:00polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/sub³
&polynomial_decay_pruning_schedule/CastCast)polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2(
&polynomial_decay_pruning_schedule/Cast
+polynomial_decay_pruning_schedule/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ¨G2-
+polynomial_decay_pruning_schedule/truediv/yä
)polynomial_decay_pruning_schedule/truedivRealDiv*polynomial_decay_pruning_schedule/Cast:y:04polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/truediv
+polynomial_decay_pruning_schedule/Maximum/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+polynomial_decay_pruning_schedule/Maximum/xç
)polynomial_decay_pruning_schedule/MaximumMaximum4polynomial_decay_pruning_schedule/Maximum/x:output:0-polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/Maximum
+polynomial_decay_pruning_schedule/Minimum/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+polynomial_decay_pruning_schedule/Minimum/xç
)polynomial_decay_pruning_schedule/MinimumMinimum4polynomial_decay_pruning_schedule/Minimum/x:output:0-polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/Minimum
)polynomial_decay_pruning_schedule/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)polynomial_decay_pruning_schedule/sub_1/xÝ
'polynomial_decay_pruning_schedule/sub_1Sub2polynomial_decay_pruning_schedule/sub_1/x:output:0-polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2)
'polynomial_decay_pruning_schedule/sub_1
'polynomial_decay_pruning_schedule/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2)
'polynomial_decay_pruning_schedule/Pow/yÕ
%polynomial_decay_pruning_schedule/PowPow+polynomial_decay_pruning_schedule/sub_1:z:00polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/Pow
'polynomial_decay_pruning_schedule/Mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¾2)
'polynomial_decay_pruning_schedule/Mul/xÓ
%polynomial_decay_pruning_schedule/MulMul0polynomial_decay_pruning_schedule/Mul/x:output:0)polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/Mul¡
,polynomial_decay_pruning_schedule/sparsity/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2.
,polynomial_decay_pruning_schedule/sparsity/yâ
*polynomial_decay_pruning_schedule/sparsityAdd)polynomial_decay_pruning_schedule/Mul:z:05polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2,
*polynomial_decay_pruning_schedule/sparsity¬
GreaterEqual/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOpb
GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqual¦
LessEqual/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp^
LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqualT
Less/xConst*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
Less/xR
Less/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0	*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd
Sub/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOpP
Sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
SubZ

FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorModT
Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1¬
pruning_ops/Abs/ReadVariableOpReadVariableOp'pruning_ops_abs_readvariableop_resource*"
_output_shapes
:*
dtype02 
pruning_ops/Abs/ReadVariableOp~
pruning_ops/AbsAbs&pruning_ops/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:2
pruning_ops/Absf
pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value	B :02
pruning_ops/Sizew
pruning_ops/CastCastpruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
pruning_ops/Castk
pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
pruning_ops/sub/x
pruning_ops/subSubpruning_ops/sub/x:output:0.polynomial_decay_pruning_schedule/sparsity:z:0*
T0*
_output_shapes
: 2
pruning_ops/subu
pruning_ops/mulMulpruning_ops/Cast:y:0pruning_ops/sub:z:0*
T0*
_output_shapes
: 2
pruning_ops/mule
pruning_ops/RoundRoundpruning_ops/mul:z:0*
T0*
_output_shapes
: 2
pruning_ops/Rounds
pruning_ops/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
pruning_ops/Maximum/y
pruning_ops/MaximumMaximumpruning_ops/Round:y:0pruning_ops/Maximum/y:output:0*
T0*
_output_shapes
: 2
pruning_ops/Maximumy
pruning_ops/Cast_1Castpruning_ops/Maximum:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
pruning_ops/Cast_1
pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
pruning_ops/Reshape/shape
pruning_ops/ReshapeReshapepruning_ops/Abs:y:0"pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes
:02
pruning_ops/Reshapej
pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value	B :02
pruning_ops/Size_1
pruning_ops/TopKV2TopKV2pruning_ops/Reshape:output:0pruning_ops/Size_1:output:0*
T0* 
_output_shapes
:0:02
pruning_ops/TopKV2l
pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
pruning_ops/sub_1/y
pruning_ops/sub_1Subpruning_ops/Cast_1:y:0pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: 2
pruning_ops/sub_1x
pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
pruning_ops/GatherV2/axisÔ
pruning_ops/GatherV2GatherV2pruning_ops/TopKV2:values:0pruning_ops/sub_1:z:0"pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: 2
pruning_ops/GatherV2¥
pruning_ops/GreaterEqualGreaterEqualpruning_ops/Abs:y:0pruning_ops/GatherV2:output:0*
T0*"
_output_shapes
:2
pruning_ops/GreaterEqual
pruning_ops/Cast_2Castpruning_ops/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
:2
pruning_ops/Cast_2
AssignVariableOpAssignVariableOpassignvariableop_resourcepruning_ops/Cast_2:y:0*
_output_shapes
 *
dtype02
AssignVariableOp
AssignVariableOp_1AssignVariableOpassignvariableop_1_resourcepruning_ops/GatherV2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1

group_depsNoOp^AssignVariableOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_depse
IdentityIdentityidentity_logicaland_1^group_deps*
T0
*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^AssignVariableOp^AssignVariableOp_1*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
::::: 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_1: 

_output_shapes
: :
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
: 
K

cond_true_319323=
9polynomial_decay_pruning_schedule_readvariableop_resource+
'pruning_ops_abs_readvariableop_resource
assignvariableop_resource
assignvariableop_1_resource
identity_logicaland_1


identity_1
¢AssignVariableOp¢AssignVariableOp_1Ö
0polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	22
0polynomial_decay_pruning_schedule/ReadVariableOp
'polynomial_decay_pruning_schedule/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'polynomial_decay_pruning_schedule/sub/yâ
%polynomial_decay_pruning_schedule/subSub8polynomial_decay_pruning_schedule/ReadVariableOp:value:00polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/sub³
&polynomial_decay_pruning_schedule/CastCast)polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2(
&polynomial_decay_pruning_schedule/Cast
+polynomial_decay_pruning_schedule/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ¨G2-
+polynomial_decay_pruning_schedule/truediv/yä
)polynomial_decay_pruning_schedule/truedivRealDiv*polynomial_decay_pruning_schedule/Cast:y:04polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/truediv
+polynomial_decay_pruning_schedule/Maximum/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+polynomial_decay_pruning_schedule/Maximum/xç
)polynomial_decay_pruning_schedule/MaximumMaximum4polynomial_decay_pruning_schedule/Maximum/x:output:0-polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/Maximum
+polynomial_decay_pruning_schedule/Minimum/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+polynomial_decay_pruning_schedule/Minimum/xç
)polynomial_decay_pruning_schedule/MinimumMinimum4polynomial_decay_pruning_schedule/Minimum/x:output:0-polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/Minimum
)polynomial_decay_pruning_schedule/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)polynomial_decay_pruning_schedule/sub_1/xÝ
'polynomial_decay_pruning_schedule/sub_1Sub2polynomial_decay_pruning_schedule/sub_1/x:output:0-polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2)
'polynomial_decay_pruning_schedule/sub_1
'polynomial_decay_pruning_schedule/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2)
'polynomial_decay_pruning_schedule/Pow/yÕ
%polynomial_decay_pruning_schedule/PowPow+polynomial_decay_pruning_schedule/sub_1:z:00polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/Pow
'polynomial_decay_pruning_schedule/Mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¾2)
'polynomial_decay_pruning_schedule/Mul/xÓ
%polynomial_decay_pruning_schedule/MulMul0polynomial_decay_pruning_schedule/Mul/x:output:0)polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/Mul¡
,polynomial_decay_pruning_schedule/sparsity/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2.
,polynomial_decay_pruning_schedule/sparsity/yâ
*polynomial_decay_pruning_schedule/sparsityAdd)polynomial_decay_pruning_schedule/Mul:z:05polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2,
*polynomial_decay_pruning_schedule/sparsity¬
GreaterEqual/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOpb
GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqual¦
LessEqual/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp^
LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqualT
Less/xConst*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
Less/xR
Less/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0	*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd
Sub/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOpP
Sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
SubZ

FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorModT
Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1ª
pruning_ops/Abs/ReadVariableOpReadVariableOp'pruning_ops_abs_readvariableop_resource* 
_output_shapes
:
*
dtype02 
pruning_ops/Abs/ReadVariableOp|
pruning_ops/AbsAbs&pruning_ops/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2
pruning_ops/Absh
pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
valueB	 :2
pruning_ops/Sizew
pruning_ops/CastCastpruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
pruning_ops/Castk
pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
pruning_ops/sub/x
pruning_ops/subSubpruning_ops/sub/x:output:0.polynomial_decay_pruning_schedule/sparsity:z:0*
T0*
_output_shapes
: 2
pruning_ops/subu
pruning_ops/mulMulpruning_ops/Cast:y:0pruning_ops/sub:z:0*
T0*
_output_shapes
: 2
pruning_ops/mule
pruning_ops/RoundRoundpruning_ops/mul:z:0*
T0*
_output_shapes
: 2
pruning_ops/Rounds
pruning_ops/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
pruning_ops/Maximum/y
pruning_ops/MaximumMaximumpruning_ops/Round:y:0pruning_ops/Maximum/y:output:0*
T0*
_output_shapes
: 2
pruning_ops/Maximumy
pruning_ops/Cast_1Castpruning_ops/Maximum:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
pruning_ops/Cast_1
pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
pruning_ops/Reshape/shape
pruning_ops/ReshapeReshapepruning_ops/Abs:y:0"pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes

:2
pruning_ops/Reshapel
pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
valueB	 :2
pruning_ops/Size_1
pruning_ops/TopKV2TopKV2pruning_ops/Reshape:output:0pruning_ops/Size_1:output:0*
T0*$
_output_shapes
::2
pruning_ops/TopKV2l
pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
pruning_ops/sub_1/y
pruning_ops/sub_1Subpruning_ops/Cast_1:y:0pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: 2
pruning_ops/sub_1x
pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
pruning_ops/GatherV2/axisÔ
pruning_ops/GatherV2GatherV2pruning_ops/TopKV2:values:0pruning_ops/sub_1:z:0"pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: 2
pruning_ops/GatherV2£
pruning_ops/GreaterEqualGreaterEqualpruning_ops/Abs:y:0pruning_ops/GatherV2:output:0*
T0* 
_output_shapes
:
2
pruning_ops/GreaterEqual
pruning_ops/Cast_2Castpruning_ops/GreaterEqual:z:0*

DstT0*

SrcT0
* 
_output_shapes
:
2
pruning_ops/Cast_2
AssignVariableOpAssignVariableOpassignvariableop_resourcepruning_ops/Cast_2:y:0*
_output_shapes
 *
dtype02
AssignVariableOp
AssignVariableOp_1AssignVariableOpassignvariableop_1_resourcepruning_ops/GatherV2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1

group_depsNoOp^AssignVariableOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_depse
IdentityIdentityidentity_logicaland_1^group_deps*
T0
*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^AssignVariableOp^AssignVariableOp_1*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
::::: 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_1: 

_output_shapes
: :
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
: 
K

cond_true_316561=
9polynomial_decay_pruning_schedule_readvariableop_resource+
'pruning_ops_abs_readvariableop_resource
assignvariableop_resource
assignvariableop_1_resource
identity_logicaland_1


identity_1
¢AssignVariableOp¢AssignVariableOp_1Ö
0polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	22
0polynomial_decay_pruning_schedule/ReadVariableOp
'polynomial_decay_pruning_schedule/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'polynomial_decay_pruning_schedule/sub/yâ
%polynomial_decay_pruning_schedule/subSub8polynomial_decay_pruning_schedule/ReadVariableOp:value:00polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/sub³
&polynomial_decay_pruning_schedule/CastCast)polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2(
&polynomial_decay_pruning_schedule/Cast
+polynomial_decay_pruning_schedule/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ¨G2-
+polynomial_decay_pruning_schedule/truediv/yä
)polynomial_decay_pruning_schedule/truedivRealDiv*polynomial_decay_pruning_schedule/Cast:y:04polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/truediv
+polynomial_decay_pruning_schedule/Maximum/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+polynomial_decay_pruning_schedule/Maximum/xç
)polynomial_decay_pruning_schedule/MaximumMaximum4polynomial_decay_pruning_schedule/Maximum/x:output:0-polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/Maximum
+polynomial_decay_pruning_schedule/Minimum/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+polynomial_decay_pruning_schedule/Minimum/xç
)polynomial_decay_pruning_schedule/MinimumMinimum4polynomial_decay_pruning_schedule/Minimum/x:output:0-polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/Minimum
)polynomial_decay_pruning_schedule/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)polynomial_decay_pruning_schedule/sub_1/xÝ
'polynomial_decay_pruning_schedule/sub_1Sub2polynomial_decay_pruning_schedule/sub_1/x:output:0-polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2)
'polynomial_decay_pruning_schedule/sub_1
'polynomial_decay_pruning_schedule/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2)
'polynomial_decay_pruning_schedule/Pow/yÕ
%polynomial_decay_pruning_schedule/PowPow+polynomial_decay_pruning_schedule/sub_1:z:00polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/Pow
'polynomial_decay_pruning_schedule/Mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¾2)
'polynomial_decay_pruning_schedule/Mul/xÓ
%polynomial_decay_pruning_schedule/MulMul0polynomial_decay_pruning_schedule/Mul/x:output:0)polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/Mul¡
,polynomial_decay_pruning_schedule/sparsity/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2.
,polynomial_decay_pruning_schedule/sparsity/yâ
*polynomial_decay_pruning_schedule/sparsityAdd)polynomial_decay_pruning_schedule/Mul:z:05polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2,
*polynomial_decay_pruning_schedule/sparsity¬
GreaterEqual/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOpb
GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqual¦
LessEqual/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp^
LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqualT
Less/xConst*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
Less/xR
Less/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0	*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd
Sub/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOpP
Sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
SubZ

FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorModT
Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1¬
pruning_ops/Abs/ReadVariableOpReadVariableOp'pruning_ops_abs_readvariableop_resource*"
_output_shapes
:*
dtype02 
pruning_ops/Abs/ReadVariableOp~
pruning_ops/AbsAbs&pruning_ops/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:2
pruning_ops/Absg
pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value
B :
2
pruning_ops/Sizew
pruning_ops/CastCastpruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
pruning_ops/Castk
pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
pruning_ops/sub/x
pruning_ops/subSubpruning_ops/sub/x:output:0.polynomial_decay_pruning_schedule/sparsity:z:0*
T0*
_output_shapes
: 2
pruning_ops/subu
pruning_ops/mulMulpruning_ops/Cast:y:0pruning_ops/sub:z:0*
T0*
_output_shapes
: 2
pruning_ops/mule
pruning_ops/RoundRoundpruning_ops/mul:z:0*
T0*
_output_shapes
: 2
pruning_ops/Rounds
pruning_ops/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
pruning_ops/Maximum/y
pruning_ops/MaximumMaximumpruning_ops/Round:y:0pruning_ops/Maximum/y:output:0*
T0*
_output_shapes
: 2
pruning_ops/Maximumy
pruning_ops/Cast_1Castpruning_ops/Maximum:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
pruning_ops/Cast_1
pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
pruning_ops/Reshape/shape
pruning_ops/ReshapeReshapepruning_ops/Abs:y:0"pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes	
:
2
pruning_ops/Reshapek
pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value
B :
2
pruning_ops/Size_1
pruning_ops/TopKV2TopKV2pruning_ops/Reshape:output:0pruning_ops/Size_1:output:0*
T0*"
_output_shapes
:
:
2
pruning_ops/TopKV2l
pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
pruning_ops/sub_1/y
pruning_ops/sub_1Subpruning_ops/Cast_1:y:0pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: 2
pruning_ops/sub_1x
pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
pruning_ops/GatherV2/axisÔ
pruning_ops/GatherV2GatherV2pruning_ops/TopKV2:values:0pruning_ops/sub_1:z:0"pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: 2
pruning_ops/GatherV2¥
pruning_ops/GreaterEqualGreaterEqualpruning_ops/Abs:y:0pruning_ops/GatherV2:output:0*
T0*"
_output_shapes
:2
pruning_ops/GreaterEqual
pruning_ops/Cast_2Castpruning_ops/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
:2
pruning_ops/Cast_2
AssignVariableOpAssignVariableOpassignvariableop_resourcepruning_ops/Cast_2:y:0*
_output_shapes
 *
dtype02
AssignVariableOp
AssignVariableOp_1AssignVariableOpassignvariableop_1_resourcepruning_ops/GatherV2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1

group_depsNoOp^AssignVariableOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_depse
IdentityIdentityidentity_logicaland_1^group_deps*
T0
*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^AssignVariableOp^AssignVariableOp_1*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
::::: 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_1: 

_output_shapes
: :
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
: 
Ò

cond_false_319324
placeholder
placeholder_1
placeholder_2
placeholder_3
identity_logicaland_1


identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOp_
IdentityIdentityidentity_logicaland_1^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
::::: : 

_output_shapes
: :
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
: 
å
u
,prune_low_magnitude_dropout_cond_true_3179755
1identity_prune_low_magnitude_dropout_logicaland_1


identity_1
6

group_depsNoOp*
_output_shapes
 2

group_deps
IdentityIdentity1identity_prune_low_magnitude_dropout_logicaland_1^group_deps*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : 

_output_shapes
: 
Ûf
È
X__inference_prune_low_magnitude_conv1d_1_layer_call_and_return_conditional_losses_318771

inputs
readvariableop_resource
cond_input_1
cond_input_2
cond_input_3#
biasadd_readvariableop_resource
identity¢AssignVariableOp¢AssignVariableOp_1¢'assert_greater_equal/Assert/AssertGuard¢condp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOpP
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
add/y\
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: 2
add
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOpA
updateNoOp^AssignVariableOp*
_output_shapes
 2
update­
#assert_greater_equal/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
: *
dtype0	2%
#assert_greater_equal/ReadVariableOpr
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
assert_greater_equal/yÅ
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2#
!assert_greater_equal/GreaterEqual{
assert_greater_equal/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
assert_greater_equal/Const
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/Const:output:0*
_output_shapes
: 2
assert_greater_equal/All
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2#
!assert_greater_equal/Assert/Const¶
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2%
#assert_greater_equal/Assert/Const_1·
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2%
#assert_greater_equal/Assert/Const_2ª
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2%
#assert_greater_equal/Assert/Const_3
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_318618*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_3186172)
'assert_greater_equal/Assert/AssertGuardÃ
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identityú
0polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	22
0polynomial_decay_pruning_schedule/ReadVariableOpÇ
'polynomial_decay_pruning_schedule/sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'polynomial_decay_pruning_schedule/sub/yâ
%polynomial_decay_pruning_schedule/subSub8polynomial_decay_pruning_schedule/ReadVariableOp:value:00polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/sub³
&polynomial_decay_pruning_schedule/CastCast)polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2(
&polynomial_decay_pruning_schedule/CastÒ
+polynomial_decay_pruning_schedule/truediv/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 * ¨G2-
+polynomial_decay_pruning_schedule/truediv/yä
)polynomial_decay_pruning_schedule/truedivRealDiv*polynomial_decay_pruning_schedule/Cast:y:04polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/truedivÒ
+polynomial_decay_pruning_schedule/Maximum/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *    2-
+polynomial_decay_pruning_schedule/Maximum/xç
)polynomial_decay_pruning_schedule/MaximumMaximum4polynomial_decay_pruning_schedule/Maximum/x:output:0-polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/MaximumÒ
+polynomial_decay_pruning_schedule/Minimum/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+polynomial_decay_pruning_schedule/Minimum/xç
)polynomial_decay_pruning_schedule/MinimumMinimum4polynomial_decay_pruning_schedule/Minimum/x:output:0-polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/MinimumÎ
)polynomial_decay_pruning_schedule/sub_1/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)polynomial_decay_pruning_schedule/sub_1/xÝ
'polynomial_decay_pruning_schedule/sub_1Sub2polynomial_decay_pruning_schedule/sub_1/x:output:0-polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2)
'polynomial_decay_pruning_schedule/sub_1Ê
'polynomial_decay_pruning_schedule/Pow/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  @@2)
'polynomial_decay_pruning_schedule/Pow/yÕ
%polynomial_decay_pruning_schedule/PowPow+polynomial_decay_pruning_schedule/sub_1:z:00polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/PowÊ
'polynomial_decay_pruning_schedule/Mul/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *¾2)
'polynomial_decay_pruning_schedule/Mul/xÓ
%polynomial_decay_pruning_schedule/MulMul0polynomial_decay_pruning_schedule/Mul/x:output:0)polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/MulÔ
,polynomial_decay_pruning_schedule/sparsity/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2.
,polynomial_decay_pruning_schedule/sparsity/yâ
*polynomial_decay_pruning_schedule/sparsityAdd)polynomial_decay_pruning_schedule/Mul:z:05polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2,
*polynomial_decay_pruning_schedule/sparsityÐ
GreaterEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOp
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqualÊ
LessEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqual
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
Less/x
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0	*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd¾
Sub/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOp
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
Sub

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorMod
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1û
condIfLogicalAnd_1:z:0readvariableop_resourcecond_input_1cond_input_2cond_input_3LogicalAnd_1:z:0^AssignVariableOp*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*$
else_branchR
cond_false_318675*
output_shapes
: *#
then_branchR
cond_true_3186742
condZ
cond/IdentityIdentitycond:output:0*
T0
*
_output_shapes
: 2
cond/Identityu
update_1NoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 2

update_1y
Mul/ReadVariableOpReadVariableOpcond_input_1*"
_output_shapes
:*
dtype02
Mul/ReadVariableOp
Mul/ReadVariableOp_1ReadVariableOpcond_input_2^cond*"
_output_shapes
:*
dtype02
Mul/ReadVariableOp_1x
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*"
_output_shapes
:2
Mul
AssignVariableOp_1AssignVariableOpcond_input_1Mul:z:0^Mul/ReadVariableOp^cond*
_output_shapes
 *
dtype02
AssignVariableOp_1y

group_depsNoOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_depsu
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2
group_deps_1p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	2
conv1d/ExpandDims®
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOpcond_input_1^AssignVariableOp_1*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1¸
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	*
squeeze_dims
2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	2
ReluÄ
IdentityIdentityRelu:activations:0^AssignVariableOp^AssignVariableOp_1(^assert_greater_equal/Assert/AssertGuard^cond*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿà	:::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2
condcond:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	
 
_user_specified_nameinputs:
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
: 
ò
¿
=__inference_prune_low_magnitude_conv1d_1_layer_call_fn_318806

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	*#
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*a
f\RZ
X__inference_prune_low_magnitude_conv1d_1_layer_call_and_return_conditional_losses_3164292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿà	:::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	
 
_user_specified_nameinputs:
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
: 

u
W__inference_prune_low_magnitude_dropout_layer_call_and_return_conditional_losses_316816

inputs

identity_14
	no_updateNoOp*
_output_shapes
 2
	no_update8
no_update_1NoOp*
_output_shapes
 2
no_update_16

group_depsNoOp*
_output_shapes
 2

group_deps_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÙ	:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	
 
_user_specified_nameinputs
K

cond_true_316103=
9polynomial_decay_pruning_schedule_readvariableop_resource+
'pruning_ops_abs_readvariableop_resource
assignvariableop_resource
assignvariableop_1_resource
identity_logicaland_1


identity_1
¢AssignVariableOp¢AssignVariableOp_1Ö
0polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	22
0polynomial_decay_pruning_schedule/ReadVariableOp
'polynomial_decay_pruning_schedule/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'polynomial_decay_pruning_schedule/sub/yâ
%polynomial_decay_pruning_schedule/subSub8polynomial_decay_pruning_schedule/ReadVariableOp:value:00polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/sub³
&polynomial_decay_pruning_schedule/CastCast)polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2(
&polynomial_decay_pruning_schedule/Cast
+polynomial_decay_pruning_schedule/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ¨G2-
+polynomial_decay_pruning_schedule/truediv/yä
)polynomial_decay_pruning_schedule/truedivRealDiv*polynomial_decay_pruning_schedule/Cast:y:04polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/truediv
+polynomial_decay_pruning_schedule/Maximum/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+polynomial_decay_pruning_schedule/Maximum/xç
)polynomial_decay_pruning_schedule/MaximumMaximum4polynomial_decay_pruning_schedule/Maximum/x:output:0-polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/Maximum
+polynomial_decay_pruning_schedule/Minimum/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+polynomial_decay_pruning_schedule/Minimum/xç
)polynomial_decay_pruning_schedule/MinimumMinimum4polynomial_decay_pruning_schedule/Minimum/x:output:0-polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/Minimum
)polynomial_decay_pruning_schedule/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)polynomial_decay_pruning_schedule/sub_1/xÝ
'polynomial_decay_pruning_schedule/sub_1Sub2polynomial_decay_pruning_schedule/sub_1/x:output:0-polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2)
'polynomial_decay_pruning_schedule/sub_1
'polynomial_decay_pruning_schedule/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2)
'polynomial_decay_pruning_schedule/Pow/yÕ
%polynomial_decay_pruning_schedule/PowPow+polynomial_decay_pruning_schedule/sub_1:z:00polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/Pow
'polynomial_decay_pruning_schedule/Mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¾2)
'polynomial_decay_pruning_schedule/Mul/xÓ
%polynomial_decay_pruning_schedule/MulMul0polynomial_decay_pruning_schedule/Mul/x:output:0)polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/Mul¡
,polynomial_decay_pruning_schedule/sparsity/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2.
,polynomial_decay_pruning_schedule/sparsity/yâ
*polynomial_decay_pruning_schedule/sparsityAdd)polynomial_decay_pruning_schedule/Mul:z:05polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2,
*polynomial_decay_pruning_schedule/sparsity¬
GreaterEqual/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOpb
GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqual¦
LessEqual/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp^
LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqualT
Less/xConst*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
Less/xR
Less/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0	*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd
Sub/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOpP
Sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
SubZ

FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorModT
Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1¬
pruning_ops/Abs/ReadVariableOpReadVariableOp'pruning_ops_abs_readvariableop_resource*"
_output_shapes
:*
dtype02 
pruning_ops/Abs/ReadVariableOp~
pruning_ops/AbsAbs&pruning_ops/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:2
pruning_ops/Absf
pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value	B :02
pruning_ops/Sizew
pruning_ops/CastCastpruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
pruning_ops/Castk
pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
pruning_ops/sub/x
pruning_ops/subSubpruning_ops/sub/x:output:0.polynomial_decay_pruning_schedule/sparsity:z:0*
T0*
_output_shapes
: 2
pruning_ops/subu
pruning_ops/mulMulpruning_ops/Cast:y:0pruning_ops/sub:z:0*
T0*
_output_shapes
: 2
pruning_ops/mule
pruning_ops/RoundRoundpruning_ops/mul:z:0*
T0*
_output_shapes
: 2
pruning_ops/Rounds
pruning_ops/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
pruning_ops/Maximum/y
pruning_ops/MaximumMaximumpruning_ops/Round:y:0pruning_ops/Maximum/y:output:0*
T0*
_output_shapes
: 2
pruning_ops/Maximumy
pruning_ops/Cast_1Castpruning_ops/Maximum:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
pruning_ops/Cast_1
pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
pruning_ops/Reshape/shape
pruning_ops/ReshapeReshapepruning_ops/Abs:y:0"pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes
:02
pruning_ops/Reshapej
pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value	B :02
pruning_ops/Size_1
pruning_ops/TopKV2TopKV2pruning_ops/Reshape:output:0pruning_ops/Size_1:output:0*
T0* 
_output_shapes
:0:02
pruning_ops/TopKV2l
pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
pruning_ops/sub_1/y
pruning_ops/sub_1Subpruning_ops/Cast_1:y:0pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: 2
pruning_ops/sub_1x
pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
pruning_ops/GatherV2/axisÔ
pruning_ops/GatherV2GatherV2pruning_ops/TopKV2:values:0pruning_ops/sub_1:z:0"pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: 2
pruning_ops/GatherV2¥
pruning_ops/GreaterEqualGreaterEqualpruning_ops/Abs:y:0pruning_ops/GatherV2:output:0*
T0*"
_output_shapes
:2
pruning_ops/GreaterEqual
pruning_ops/Cast_2Castpruning_ops/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
:2
pruning_ops/Cast_2
AssignVariableOpAssignVariableOpassignvariableop_resourcepruning_ops/Cast_2:y:0*
_output_shapes
 *
dtype02
AssignVariableOp
AssignVariableOp_1AssignVariableOpassignvariableop_1_resourcepruning_ops/GatherV2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1

group_depsNoOp^AssignVariableOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_depse
IdentityIdentityidentity_logicaland_1^group_deps*
T0
*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^AssignVariableOp^AssignVariableOp_1*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
::::: 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_1: 

_output_shapes
: :
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
: 
K

cond_true_317017=
9polynomial_decay_pruning_schedule_readvariableop_resource+
'pruning_ops_abs_readvariableop_resource
assignvariableop_resource
assignvariableop_1_resource
identity_logicaland_1


identity_1
¢AssignVariableOp¢AssignVariableOp_1Ö
0polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	22
0polynomial_decay_pruning_schedule/ReadVariableOp
'polynomial_decay_pruning_schedule/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'polynomial_decay_pruning_schedule/sub/yâ
%polynomial_decay_pruning_schedule/subSub8polynomial_decay_pruning_schedule/ReadVariableOp:value:00polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/sub³
&polynomial_decay_pruning_schedule/CastCast)polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2(
&polynomial_decay_pruning_schedule/Cast
+polynomial_decay_pruning_schedule/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ¨G2-
+polynomial_decay_pruning_schedule/truediv/yä
)polynomial_decay_pruning_schedule/truedivRealDiv*polynomial_decay_pruning_schedule/Cast:y:04polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/truediv
+polynomial_decay_pruning_schedule/Maximum/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+polynomial_decay_pruning_schedule/Maximum/xç
)polynomial_decay_pruning_schedule/MaximumMaximum4polynomial_decay_pruning_schedule/Maximum/x:output:0-polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/Maximum
+polynomial_decay_pruning_schedule/Minimum/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+polynomial_decay_pruning_schedule/Minimum/xç
)polynomial_decay_pruning_schedule/MinimumMinimum4polynomial_decay_pruning_schedule/Minimum/x:output:0-polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/Minimum
)polynomial_decay_pruning_schedule/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)polynomial_decay_pruning_schedule/sub_1/xÝ
'polynomial_decay_pruning_schedule/sub_1Sub2polynomial_decay_pruning_schedule/sub_1/x:output:0-polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2)
'polynomial_decay_pruning_schedule/sub_1
'polynomial_decay_pruning_schedule/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2)
'polynomial_decay_pruning_schedule/Pow/yÕ
%polynomial_decay_pruning_schedule/PowPow+polynomial_decay_pruning_schedule/sub_1:z:00polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/Pow
'polynomial_decay_pruning_schedule/Mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¾2)
'polynomial_decay_pruning_schedule/Mul/xÓ
%polynomial_decay_pruning_schedule/MulMul0polynomial_decay_pruning_schedule/Mul/x:output:0)polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/Mul¡
,polynomial_decay_pruning_schedule/sparsity/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2.
,polynomial_decay_pruning_schedule/sparsity/yâ
*polynomial_decay_pruning_schedule/sparsityAdd)polynomial_decay_pruning_schedule/Mul:z:05polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2,
*polynomial_decay_pruning_schedule/sparsity¬
GreaterEqual/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOpb
GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqual¦
LessEqual/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp^
LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqualT
Less/xConst*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
Less/xR
Less/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0	*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd
Sub/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOpP
Sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
SubZ

FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorModT
Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1ª
pruning_ops/Abs/ReadVariableOpReadVariableOp'pruning_ops_abs_readvariableop_resource* 
_output_shapes
:
*
dtype02 
pruning_ops/Abs/ReadVariableOp|
pruning_ops/AbsAbs&pruning_ops/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2
pruning_ops/Absh
pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
valueB	 :2
pruning_ops/Sizew
pruning_ops/CastCastpruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
pruning_ops/Castk
pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
pruning_ops/sub/x
pruning_ops/subSubpruning_ops/sub/x:output:0.polynomial_decay_pruning_schedule/sparsity:z:0*
T0*
_output_shapes
: 2
pruning_ops/subu
pruning_ops/mulMulpruning_ops/Cast:y:0pruning_ops/sub:z:0*
T0*
_output_shapes
: 2
pruning_ops/mule
pruning_ops/RoundRoundpruning_ops/mul:z:0*
T0*
_output_shapes
: 2
pruning_ops/Rounds
pruning_ops/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
pruning_ops/Maximum/y
pruning_ops/MaximumMaximumpruning_ops/Round:y:0pruning_ops/Maximum/y:output:0*
T0*
_output_shapes
: 2
pruning_ops/Maximumy
pruning_ops/Cast_1Castpruning_ops/Maximum:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
pruning_ops/Cast_1
pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
pruning_ops/Reshape/shape
pruning_ops/ReshapeReshapepruning_ops/Abs:y:0"pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes

:2
pruning_ops/Reshapel
pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
valueB	 :2
pruning_ops/Size_1
pruning_ops/TopKV2TopKV2pruning_ops/Reshape:output:0pruning_ops/Size_1:output:0*
T0*$
_output_shapes
::2
pruning_ops/TopKV2l
pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
pruning_ops/sub_1/y
pruning_ops/sub_1Subpruning_ops/Cast_1:y:0pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: 2
pruning_ops/sub_1x
pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
pruning_ops/GatherV2/axisÔ
pruning_ops/GatherV2GatherV2pruning_ops/TopKV2:values:0pruning_ops/sub_1:z:0"pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: 2
pruning_ops/GatherV2£
pruning_ops/GreaterEqualGreaterEqualpruning_ops/Abs:y:0pruning_ops/GatherV2:output:0*
T0* 
_output_shapes
:
2
pruning_ops/GreaterEqual
pruning_ops/Cast_2Castpruning_ops/GreaterEqual:z:0*

DstT0*

SrcT0
* 
_output_shapes
:
2
pruning_ops/Cast_2
AssignVariableOpAssignVariableOpassignvariableop_resourcepruning_ops/Cast_2:y:0*
_output_shapes
 *
dtype02
AssignVariableOp
AssignVariableOp_1AssignVariableOpassignvariableop_1_resourcepruning_ops/GatherV2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1

group_depsNoOp^AssignVariableOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_depse
IdentityIdentityidentity_logicaland_1^group_deps*
T0
*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^AssignVariableOp^AssignVariableOp_1*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
::::: 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_1: 

_output_shapes
: :
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
: 
ã

:__inference_prune_low_magnitude_dense_layer_call_fn_319456

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*^
fYRW
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_3171242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
éZ
Ð
W__inference_prune_low_magnitude_dropout_layer_call_and_return_conditional_losses_319127

inputs
readvariableop_resource
identity¢AssignVariableOp¢'assert_greater_equal/Assert/AssertGuardp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOpP
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
add/y\
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: 2
add
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOpA
updateNoOp^AssignVariableOp*
_output_shapes
 2
update­
#assert_greater_equal/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
: *
dtype0	2%
#assert_greater_equal/ReadVariableOpr
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
assert_greater_equal/yÅ
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2#
!assert_greater_equal/GreaterEqual{
assert_greater_equal/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
assert_greater_equal/Const
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/Const:output:0*
_output_shapes
: 2
assert_greater_equal/All
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2#
!assert_greater_equal/Assert/Const¶
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2%
#assert_greater_equal/Assert/Const_1·
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2%
#assert_greater_equal/Assert/Const_2ª
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2%
#assert_greater_equal/Assert/Const_3
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_319050*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_3190492)
'assert_greater_equal/Assert/AssertGuardÃ
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identityú
0polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	22
0polynomial_decay_pruning_schedule/ReadVariableOpÇ
'polynomial_decay_pruning_schedule/sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'polynomial_decay_pruning_schedule/sub/yâ
%polynomial_decay_pruning_schedule/subSub8polynomial_decay_pruning_schedule/ReadVariableOp:value:00polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/sub³
&polynomial_decay_pruning_schedule/CastCast)polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2(
&polynomial_decay_pruning_schedule/CastÒ
+polynomial_decay_pruning_schedule/truediv/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 * ¨G2-
+polynomial_decay_pruning_schedule/truediv/yä
)polynomial_decay_pruning_schedule/truedivRealDiv*polynomial_decay_pruning_schedule/Cast:y:04polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/truedivÒ
+polynomial_decay_pruning_schedule/Maximum/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *    2-
+polynomial_decay_pruning_schedule/Maximum/xç
)polynomial_decay_pruning_schedule/MaximumMaximum4polynomial_decay_pruning_schedule/Maximum/x:output:0-polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/MaximumÒ
+polynomial_decay_pruning_schedule/Minimum/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+polynomial_decay_pruning_schedule/Minimum/xç
)polynomial_decay_pruning_schedule/MinimumMinimum4polynomial_decay_pruning_schedule/Minimum/x:output:0-polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/MinimumÎ
)polynomial_decay_pruning_schedule/sub_1/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)polynomial_decay_pruning_schedule/sub_1/xÝ
'polynomial_decay_pruning_schedule/sub_1Sub2polynomial_decay_pruning_schedule/sub_1/x:output:0-polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2)
'polynomial_decay_pruning_schedule/sub_1Ê
'polynomial_decay_pruning_schedule/Pow/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  @@2)
'polynomial_decay_pruning_schedule/Pow/yÕ
%polynomial_decay_pruning_schedule/PowPow+polynomial_decay_pruning_schedule/sub_1:z:00polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/PowÊ
'polynomial_decay_pruning_schedule/Mul/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *¾2)
'polynomial_decay_pruning_schedule/Mul/xÓ
%polynomial_decay_pruning_schedule/MulMul0polynomial_decay_pruning_schedule/Mul/x:output:0)polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/MulÔ
,polynomial_decay_pruning_schedule/sparsity/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2.
,polynomial_decay_pruning_schedule/sparsity/yâ
*polynomial_decay_pruning_schedule/sparsityAdd)polynomial_decay_pruning_schedule/Mul:z:05polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2,
*polynomial_decay_pruning_schedule/sparsityÐ
GreaterEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOp
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqualÊ
LessEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqual
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
Less/x
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0	*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd¾
Sub/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOp
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
Sub

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorMod
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1¦
condStatelessIfLogicalAnd_1:z:0LogicalAnd_1:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *$
else_branchR
cond_false_319107*
output_shapes
: *#
then_branchR
cond_true_3191062
condZ
cond/IdentityIdentitycond:output:0*
T0
*
_output_shapes
: 2
cond/Identityu
update_1NoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 2

update_16

group_depsNoOp*
_output_shapes
 2

group_depsc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/GreaterEqual/yÃ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2
dropout/Mul_1§
IdentityIdentitydropout/Mul_1:z:0^AssignVariableOp(^assert_greater_equal/Assert/AssertGuard*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÙ	:2$
AssignVariableOpAssignVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	
 
_user_specified_nameinputs:

_output_shapes
: 
î
½
;__inference_prune_low_magnitude_conv1d_layer_call_fn_318590

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	*#
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*_
fZRX
V__inference_prune_low_magnitude_conv1d_layer_call_and_return_conditional_losses_3162002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿâ	:::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	
 
_user_specified_nameinputs:
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
: 
ã
Ê
4assert_greater_equal_Assert_AssertGuard_false_316047#
assert_assert_greater_equal_all
.
*assert_assert_greater_equal_readvariableop	!
assert_assert_greater_equal_y	

identity_1
¢Assertî
Assert/data_0Const*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2
Assert/data_0
Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2
Assert/data_1
Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2
Assert/data_2~
Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2
Assert/data_4
AssertAssertassert_assert_greater_equal_allAssert/data_0:output:0Assert/data_1:output:0Assert/data_2:output:0*assert_assert_greater_equal_readvariableopAssert/data_4:output:0assert_assert_greater_equal_y*
T

2		*
_output_shapes
 2
Assertk
IdentityIdentityassert_assert_greater_equal_all^Assert*
T0
*
_output_shapes
: 2

Identitya

Identity_1IdentityIdentity:output:0^Assert*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : 2
AssertAssert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ò

cond_false_318675
placeholder
placeholder_1
placeholder_2
placeholder_3
identity_logicaland_1


identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOp_
IdentityIdentityidentity_logicaland_1^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
::::: : 

_output_shapes
: :
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
: 


3assert_greater_equal_Assert_AssertGuard_true_319049%
!identity_assert_greater_equal_all

placeholder	
placeholder_1	

identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOpk
IdentityIdentity!identity_assert_greater_equal_all^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¾
ì
F__inference_sequential_layer_call_and_return_conditional_losses_318241

inputs6
2prune_low_magnitude_conv1d_readvariableop_resource+
'prune_low_magnitude_conv1d_cond_input_1+
'prune_low_magnitude_conv1d_cond_input_2+
'prune_low_magnitude_conv1d_cond_input_3>
:prune_low_magnitude_conv1d_biasadd_readvariableop_resource8
4prune_low_magnitude_conv1d_1_readvariableop_resource-
)prune_low_magnitude_conv1d_1_cond_input_1-
)prune_low_magnitude_conv1d_1_cond_input_2-
)prune_low_magnitude_conv1d_1_cond_input_3@
<prune_low_magnitude_conv1d_1_biasadd_readvariableop_resource8
4prune_low_magnitude_conv1d_2_readvariableop_resource-
)prune_low_magnitude_conv1d_2_cond_input_1-
)prune_low_magnitude_conv1d_2_cond_input_2-
)prune_low_magnitude_conv1d_2_cond_input_3@
<prune_low_magnitude_conv1d_2_biasadd_readvariableop_resource7
3prune_low_magnitude_dropout_readvariableop_resource7
3prune_low_magnitude_flatten_readvariableop_resource5
1prune_low_magnitude_dense_readvariableop_resource*
&prune_low_magnitude_dense_cond_input_1*
&prune_low_magnitude_dense_cond_input_2*
&prune_low_magnitude_dense_cond_input_3=
9prune_low_magnitude_dense_biasadd_readvariableop_resource
identity¢+prune_low_magnitude_conv1d/AssignVariableOp¢-prune_low_magnitude_conv1d/AssignVariableOp_1¢Bprune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard¢prune_low_magnitude_conv1d/cond¢-prune_low_magnitude_conv1d_1/AssignVariableOp¢/prune_low_magnitude_conv1d_1/AssignVariableOp_1¢Dprune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard¢!prune_low_magnitude_conv1d_1/cond¢-prune_low_magnitude_conv1d_2/AssignVariableOp¢/prune_low_magnitude_conv1d_2/AssignVariableOp_1¢Dprune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard¢!prune_low_magnitude_conv1d_2/cond¢*prune_low_magnitude_dense/AssignVariableOp¢,prune_low_magnitude_dense/AssignVariableOp_1¢Aprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard¢prune_low_magnitude_dense/cond¢,prune_low_magnitude_dropout/AssignVariableOp¢Cprune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard¢,prune_low_magnitude_flatten/AssignVariableOp¢Cprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuardÁ
)prune_low_magnitude_conv1d/ReadVariableOpReadVariableOp2prune_low_magnitude_conv1d_readvariableop_resource*
_output_shapes
: *
dtype0	2+
)prune_low_magnitude_conv1d/ReadVariableOp
 prune_low_magnitude_conv1d/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2"
 prune_low_magnitude_conv1d/add/yÈ
prune_low_magnitude_conv1d/addAddV21prune_low_magnitude_conv1d/ReadVariableOp:value:0)prune_low_magnitude_conv1d/add/y:output:0*
T0	*
_output_shapes
: 2 
prune_low_magnitude_conv1d/add
+prune_low_magnitude_conv1d/AssignVariableOpAssignVariableOp2prune_low_magnitude_conv1d_readvariableop_resource"prune_low_magnitude_conv1d/add:z:0*^prune_low_magnitude_conv1d/ReadVariableOp*
_output_shapes
 *
dtype0	2-
+prune_low_magnitude_conv1d/AssignVariableOp
!prune_low_magnitude_conv1d/updateNoOp,^prune_low_magnitude_conv1d/AssignVariableOp*
_output_shapes
 2#
!prune_low_magnitude_conv1d/update
>prune_low_magnitude_conv1d/assert_greater_equal/ReadVariableOpReadVariableOp2prune_low_magnitude_conv1d_readvariableop_resource,^prune_low_magnitude_conv1d/AssignVariableOp*
_output_shapes
: *
dtype0	2@
>prune_low_magnitude_conv1d/assert_greater_equal/ReadVariableOp¨
1prune_low_magnitude_conv1d/assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R23
1prune_low_magnitude_conv1d/assert_greater_equal/y±
<prune_low_magnitude_conv1d/assert_greater_equal/GreaterEqualGreaterEqualFprune_low_magnitude_conv1d/assert_greater_equal/ReadVariableOp:value:0:prune_low_magnitude_conv1d/assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2>
<prune_low_magnitude_conv1d/assert_greater_equal/GreaterEqual±
5prune_low_magnitude_conv1d/assert_greater_equal/ConstConst*
_output_shapes
: *
dtype0*
valueB 27
5prune_low_magnitude_conv1d/assert_greater_equal/Const
3prune_low_magnitude_conv1d/assert_greater_equal/AllAll@prune_low_magnitude_conv1d/assert_greater_equal/GreaterEqual:z:0>prune_low_magnitude_conv1d/assert_greater_equal/Const:output:0*
_output_shapes
: 25
3prune_low_magnitude_conv1d/assert_greater_equal/AllÌ
<prune_low_magnitude_conv1d/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2>
<prune_low_magnitude_conv1d/assert_greater_equal/Assert/Constì
>prune_low_magnitude_conv1d/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2@
>prune_low_magnitude_conv1d/assert_greater_equal/Assert/Const_1
>prune_low_magnitude_conv1d/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*X
valueOBM BGx (prune_low_magnitude_conv1d/assert_greater_equal/ReadVariableOp:0) = 2@
>prune_low_magnitude_conv1d/assert_greater_equal/Assert/Const_2û
>prune_low_magnitude_conv1d/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*K
valueBB@ B:y (prune_low_magnitude_conv1d/assert_greater_equal/y:0) = 2@
>prune_low_magnitude_conv1d/assert_greater_equal/Assert/Const_3ó
Bprune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuardIf<prune_low_magnitude_conv1d/assert_greater_equal/All:output:0<prune_low_magnitude_conv1d/assert_greater_equal/All:output:0Fprune_low_magnitude_conv1d/assert_greater_equal/ReadVariableOp:value:0:prune_low_magnitude_conv1d/assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *b
else_branchSRQ
Oprune_low_magnitude_conv1d_assert_greater_equal_Assert_AssertGuard_false_317421*
output_shapes
: *a
then_branchRRP
Nprune_low_magnitude_conv1d_assert_greater_equal_Assert_AssertGuard_true_3174202D
Bprune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard
Kprune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard/IdentityIdentityKprune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 2M
Kprune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard/Identity
Kprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOp2prune_low_magnitude_conv1d_readvariableop_resource,^prune_low_magnitude_conv1d/AssignVariableOpL^prune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2M
Kprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/ReadVariableOp
Bprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/sub/yConstL^prune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2D
Bprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/sub/yÎ
@prune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/subSubSprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/ReadVariableOp:value:0Kprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2B
@prune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/sub
Aprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/CastCastDprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2C
Aprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/Cast£
Fprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/truediv/yConstL^prune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 * ¨G2H
Fprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/truediv/yÐ
Dprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/truedivRealDivEprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/Cast:y:0Oprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2F
Dprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/truediv£
Fprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/Maximum/xConstL^prune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *    2H
Fprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/Maximum/xÓ
Dprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/MaximumMaximumOprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/Maximum/x:output:0Hprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2F
Dprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/Maximum£
Fprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/Minimum/xConstL^prune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2H
Fprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/Minimum/xÓ
Dprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/MinimumMinimumOprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/Minimum/x:output:0Hprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2F
Dprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/Minimum
Dprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/sub_1/xConstL^prune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2F
Dprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/sub_1/xÉ
Bprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/sub_1SubMprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/sub_1/x:output:0Hprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2D
Bprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/sub_1
Bprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/Pow/yConstL^prune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  @@2D
Bprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/Pow/yÁ
@prune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/PowPowFprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/sub_1:z:0Kprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2B
@prune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/Pow
Bprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/Mul/xConstL^prune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *¾2D
Bprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/Mul/x¿
@prune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/MulMulKprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/Mul/x:output:0Dprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2B
@prune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/Mul¥
Gprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/sparsity/yConstL^prune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2I
Gprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/sparsity/yÎ
Eprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/sparsityAddDprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/Mul:z:0Pprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2G
Eprune_low_magnitude_conv1d/polynomial_decay_pruning_schedule/sparsity×
6prune_low_magnitude_conv1d/GreaterEqual/ReadVariableOpReadVariableOp2prune_low_magnitude_conv1d_readvariableop_resource,^prune_low_magnitude_conv1d/AssignVariableOpL^prune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	28
6prune_low_magnitude_conv1d/GreaterEqual/ReadVariableOpæ
)prune_low_magnitude_conv1d/GreaterEqual/yConstL^prune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2+
)prune_low_magnitude_conv1d/GreaterEqual/y÷
'prune_low_magnitude_conv1d/GreaterEqualGreaterEqual>prune_low_magnitude_conv1d/GreaterEqual/ReadVariableOp:value:02prune_low_magnitude_conv1d/GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2)
'prune_low_magnitude_conv1d/GreaterEqualÑ
3prune_low_magnitude_conv1d/LessEqual/ReadVariableOpReadVariableOp2prune_low_magnitude_conv1d_readvariableop_resource,^prune_low_magnitude_conv1d/AssignVariableOpL^prune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	25
3prune_low_magnitude_conv1d/LessEqual/ReadVariableOpâ
&prune_low_magnitude_conv1d/LessEqual/yConstL^prune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2(
&prune_low_magnitude_conv1d/LessEqual/yè
$prune_low_magnitude_conv1d/LessEqual	LessEqual;prune_low_magnitude_conv1d/LessEqual/ReadVariableOp:value:0/prune_low_magnitude_conv1d/LessEqual/y:output:0*
T0	*
_output_shapes
: 2&
$prune_low_magnitude_conv1d/LessEqualØ
!prune_low_magnitude_conv1d/Less/xConstL^prune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2#
!prune_low_magnitude_conv1d/Less/xÖ
!prune_low_magnitude_conv1d/Less/yConstL^prune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2#
!prune_low_magnitude_conv1d/Less/yÃ
prune_low_magnitude_conv1d/LessLess*prune_low_magnitude_conv1d/Less/x:output:0*prune_low_magnitude_conv1d/Less/y:output:0*
T0	*
_output_shapes
: 2!
prune_low_magnitude_conv1d/LessÀ
$prune_low_magnitude_conv1d/LogicalOr	LogicalOr(prune_low_magnitude_conv1d/LessEqual:z:0#prune_low_magnitude_conv1d/Less:z:0*
_output_shapes
: 2&
$prune_low_magnitude_conv1d/LogicalOrË
%prune_low_magnitude_conv1d/LogicalAnd
LogicalAnd+prune_low_magnitude_conv1d/GreaterEqual:z:0(prune_low_magnitude_conv1d/LogicalOr:z:0*
_output_shapes
: 2'
%prune_low_magnitude_conv1d/LogicalAndÅ
-prune_low_magnitude_conv1d/Sub/ReadVariableOpReadVariableOp2prune_low_magnitude_conv1d_readvariableop_resource,^prune_low_magnitude_conv1d/AssignVariableOpL^prune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2/
-prune_low_magnitude_conv1d/Sub/ReadVariableOpÔ
 prune_low_magnitude_conv1d/Sub/yConstL^prune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2"
 prune_low_magnitude_conv1d/Sub/yÊ
prune_low_magnitude_conv1d/SubSub5prune_low_magnitude_conv1d/Sub/ReadVariableOp:value:0)prune_low_magnitude_conv1d/Sub/y:output:0*
T0	*
_output_shapes
: 2 
prune_low_magnitude_conv1d/SubÞ
%prune_low_magnitude_conv1d/FloorMod/yConstL^prune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2'
%prune_low_magnitude_conv1d/FloorMod/yË
#prune_low_magnitude_conv1d/FloorModFloorMod"prune_low_magnitude_conv1d/Sub:z:0.prune_low_magnitude_conv1d/FloorMod/y:output:0*
T0	*
_output_shapes
: 2%
#prune_low_magnitude_conv1d/FloorModØ
"prune_low_magnitude_conv1d/Equal/yConstL^prune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2$
"prune_low_magnitude_conv1d/Equal/yÄ
 prune_low_magnitude_conv1d/EqualEqual'prune_low_magnitude_conv1d/FloorMod:z:0+prune_low_magnitude_conv1d/Equal/y:output:0*
T0	*
_output_shapes
: 2"
 prune_low_magnitude_conv1d/EqualÉ
'prune_low_magnitude_conv1d/LogicalAnd_1
LogicalAnd)prune_low_magnitude_conv1d/LogicalAnd:z:0$prune_low_magnitude_conv1d/Equal:z:0*
_output_shapes
: 2)
'prune_low_magnitude_conv1d/LogicalAnd_1¤
prune_low_magnitude_conv1d/condIf+prune_low_magnitude_conv1d/LogicalAnd_1:z:02prune_low_magnitude_conv1d_readvariableop_resource'prune_low_magnitude_conv1d_cond_input_1'prune_low_magnitude_conv1d_cond_input_2'prune_low_magnitude_conv1d_cond_input_3+prune_low_magnitude_conv1d/LogicalAnd_1:z:0,^prune_low_magnitude_conv1d/AssignVariableOp*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*?
else_branch0R.
,prune_low_magnitude_conv1d_cond_false_317478*
output_shapes
: *>
then_branch/R-
+prune_low_magnitude_conv1d_cond_true_3174772!
prune_low_magnitude_conv1d/cond«
(prune_low_magnitude_conv1d/cond/IdentityIdentity(prune_low_magnitude_conv1d/cond:output:0*
T0
*
_output_shapes
: 2*
(prune_low_magnitude_conv1d/cond/Identityá
#prune_low_magnitude_conv1d/update_1NoOpL^prune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard/Identity)^prune_low_magnitude_conv1d/cond/Identity*
_output_shapes
 2%
#prune_low_magnitude_conv1d/update_1Ê
-prune_low_magnitude_conv1d/Mul/ReadVariableOpReadVariableOp'prune_low_magnitude_conv1d_cond_input_1*"
_output_shapes
:*
dtype02/
-prune_low_magnitude_conv1d/Mul/ReadVariableOpð
/prune_low_magnitude_conv1d/Mul/ReadVariableOp_1ReadVariableOp'prune_low_magnitude_conv1d_cond_input_2 ^prune_low_magnitude_conv1d/cond*"
_output_shapes
:*
dtype021
/prune_low_magnitude_conv1d/Mul/ReadVariableOp_1ä
prune_low_magnitude_conv1d/MulMul5prune_low_magnitude_conv1d/Mul/ReadVariableOp:value:07prune_low_magnitude_conv1d/Mul/ReadVariableOp_1:value:0*
T0*"
_output_shapes
:2 
prune_low_magnitude_conv1d/Mul´
-prune_low_magnitude_conv1d/AssignVariableOp_1AssignVariableOp'prune_low_magnitude_conv1d_cond_input_1"prune_low_magnitude_conv1d/Mul:z:0.^prune_low_magnitude_conv1d/Mul/ReadVariableOp ^prune_low_magnitude_conv1d/cond*
_output_shapes
 *
dtype02/
-prune_low_magnitude_conv1d/AssignVariableOp_1Ê
%prune_low_magnitude_conv1d/group_depsNoOp.^prune_low_magnitude_conv1d/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2'
%prune_low_magnitude_conv1d/group_depsÆ
'prune_low_magnitude_conv1d/group_deps_1NoOp&^prune_low_magnitude_conv1d/group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2)
'prune_low_magnitude_conv1d/group_deps_1¦
0prune_low_magnitude_conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :22
0prune_low_magnitude_conv1d/conv1d/ExpandDims/dimè
,prune_low_magnitude_conv1d/conv1d/ExpandDims
ExpandDimsinputs9prune_low_magnitude_conv1d/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	2.
,prune_low_magnitude_conv1d/conv1d/ExpandDims
=prune_low_magnitude_conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp'prune_low_magnitude_conv1d_cond_input_1.^prune_low_magnitude_conv1d/AssignVariableOp_1*"
_output_shapes
:*
dtype02?
=prune_low_magnitude_conv1d/conv1d/ExpandDims_1/ReadVariableOpª
2prune_low_magnitude_conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 24
2prune_low_magnitude_conv1d/conv1d/ExpandDims_1/dim£
.prune_low_magnitude_conv1d/conv1d/ExpandDims_1
ExpandDimsEprune_low_magnitude_conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0;prune_low_magnitude_conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:20
.prune_low_magnitude_conv1d/conv1d/ExpandDims_1¤
!prune_low_magnitude_conv1d/conv1dConv2D5prune_low_magnitude_conv1d/conv1d/ExpandDims:output:07prune_low_magnitude_conv1d/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	*
paddingVALID*
strides
2#
!prune_low_magnitude_conv1d/conv1dÛ
)prune_low_magnitude_conv1d/conv1d/SqueezeSqueeze*prune_low_magnitude_conv1d/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	*
squeeze_dims
2+
)prune_low_magnitude_conv1d/conv1d/SqueezeÝ
1prune_low_magnitude_conv1d/BiasAdd/ReadVariableOpReadVariableOp:prune_low_magnitude_conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1prune_low_magnitude_conv1d/BiasAdd/ReadVariableOpù
"prune_low_magnitude_conv1d/BiasAddBiasAdd2prune_low_magnitude_conv1d/conv1d/Squeeze:output:09prune_low_magnitude_conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	2$
"prune_low_magnitude_conv1d/BiasAdd®
prune_low_magnitude_conv1d/ReluRelu+prune_low_magnitude_conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	2!
prune_low_magnitude_conv1d/ReluÇ
+prune_low_magnitude_conv1d_1/ReadVariableOpReadVariableOp4prune_low_magnitude_conv1d_1_readvariableop_resource*
_output_shapes
: *
dtype0	2-
+prune_low_magnitude_conv1d_1/ReadVariableOp
"prune_low_magnitude_conv1d_1/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"prune_low_magnitude_conv1d_1/add/yÐ
 prune_low_magnitude_conv1d_1/addAddV23prune_low_magnitude_conv1d_1/ReadVariableOp:value:0+prune_low_magnitude_conv1d_1/add/y:output:0*
T0	*
_output_shapes
: 2"
 prune_low_magnitude_conv1d_1/add
-prune_low_magnitude_conv1d_1/AssignVariableOpAssignVariableOp4prune_low_magnitude_conv1d_1_readvariableop_resource$prune_low_magnitude_conv1d_1/add:z:0,^prune_low_magnitude_conv1d_1/ReadVariableOp*
_output_shapes
 *
dtype0	2/
-prune_low_magnitude_conv1d_1/AssignVariableOp
#prune_low_magnitude_conv1d_1/updateNoOp.^prune_low_magnitude_conv1d_1/AssignVariableOp*
_output_shapes
 2%
#prune_low_magnitude_conv1d_1/update¡
@prune_low_magnitude_conv1d_1/assert_greater_equal/ReadVariableOpReadVariableOp4prune_low_magnitude_conv1d_1_readvariableop_resource.^prune_low_magnitude_conv1d_1/AssignVariableOp*
_output_shapes
: *
dtype0	2B
@prune_low_magnitude_conv1d_1/assert_greater_equal/ReadVariableOp¬
3prune_low_magnitude_conv1d_1/assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R25
3prune_low_magnitude_conv1d_1/assert_greater_equal/y¹
>prune_low_magnitude_conv1d_1/assert_greater_equal/GreaterEqualGreaterEqualHprune_low_magnitude_conv1d_1/assert_greater_equal/ReadVariableOp:value:0<prune_low_magnitude_conv1d_1/assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2@
>prune_low_magnitude_conv1d_1/assert_greater_equal/GreaterEqualµ
7prune_low_magnitude_conv1d_1/assert_greater_equal/ConstConst*
_output_shapes
: *
dtype0*
valueB 29
7prune_low_magnitude_conv1d_1/assert_greater_equal/Const
5prune_low_magnitude_conv1d_1/assert_greater_equal/AllAllBprune_low_magnitude_conv1d_1/assert_greater_equal/GreaterEqual:z:0@prune_low_magnitude_conv1d_1/assert_greater_equal/Const:output:0*
_output_shapes
: 27
5prune_low_magnitude_conv1d_1/assert_greater_equal/AllÐ
>prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2@
>prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/Constð
@prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2B
@prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/Const_1
@prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*Z
valueQBO BIx (prune_low_magnitude_conv1d_1/assert_greater_equal/ReadVariableOp:0) = 2B
@prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/Const_2
@prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*M
valueDBB B<y (prune_low_magnitude_conv1d_1/assert_greater_equal/y:0) = 2B
@prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/Const_3È
Dprune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuardIf>prune_low_magnitude_conv1d_1/assert_greater_equal/All:output:0>prune_low_magnitude_conv1d_1/assert_greater_equal/All:output:0Hprune_low_magnitude_conv1d_1/assert_greater_equal/ReadVariableOp:value:0<prune_low_magnitude_conv1d_1/assert_greater_equal/y:output:0C^prune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *d
else_branchURS
Qprune_low_magnitude_conv1d_1_assert_greater_equal_Assert_AssertGuard_false_317587*
output_shapes
: *c
then_branchTRR
Pprune_low_magnitude_conv1d_1_assert_greater_equal_Assert_AssertGuard_true_3175862F
Dprune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard
Mprune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard/IdentityIdentityMprune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 2O
Mprune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard/Identity
Mprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOp4prune_low_magnitude_conv1d_1_readvariableop_resource.^prune_low_magnitude_conv1d_1/AssignVariableOpN^prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2O
Mprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/ReadVariableOp
Dprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/sub/yConstN^prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2F
Dprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/sub/yÖ
Bprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/subSubUprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/ReadVariableOp:value:0Mprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2D
Bprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/sub
Cprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/CastCastFprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2E
Cprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/Cast©
Hprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/truediv/yConstN^prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 * ¨G2J
Hprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/truediv/yØ
Fprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/truedivRealDivGprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/Cast:y:0Qprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2H
Fprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/truediv©
Hprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/Maximum/xConstN^prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *    2J
Hprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/Maximum/xÛ
Fprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/MaximumMaximumQprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/Maximum/x:output:0Jprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2H
Fprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/Maximum©
Hprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/Minimum/xConstN^prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2J
Hprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/Minimum/xÛ
Fprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/MinimumMinimumQprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/Minimum/x:output:0Jprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2H
Fprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/Minimum¥
Fprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/sub_1/xConstN^prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2H
Fprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/sub_1/xÑ
Dprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/sub_1SubOprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/sub_1/x:output:0Jprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2F
Dprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/sub_1¡
Dprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/Pow/yConstN^prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  @@2F
Dprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/Pow/yÉ
Bprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/PowPowHprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/sub_1:z:0Mprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2D
Bprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/Pow¡
Dprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/Mul/xConstN^prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *¾2F
Dprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/Mul/xÇ
Bprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/MulMulMprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/Mul/x:output:0Fprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2D
Bprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/Mul«
Iprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/sparsity/yConstN^prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2K
Iprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/sparsity/yÖ
Gprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/sparsityAddFprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/Mul:z:0Rprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2I
Gprune_low_magnitude_conv1d_1/polynomial_decay_pruning_schedule/sparsityá
8prune_low_magnitude_conv1d_1/GreaterEqual/ReadVariableOpReadVariableOp4prune_low_magnitude_conv1d_1_readvariableop_resource.^prune_low_magnitude_conv1d_1/AssignVariableOpN^prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2:
8prune_low_magnitude_conv1d_1/GreaterEqual/ReadVariableOpì
+prune_low_magnitude_conv1d_1/GreaterEqual/yConstN^prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2-
+prune_low_magnitude_conv1d_1/GreaterEqual/yÿ
)prune_low_magnitude_conv1d_1/GreaterEqualGreaterEqual@prune_low_magnitude_conv1d_1/GreaterEqual/ReadVariableOp:value:04prune_low_magnitude_conv1d_1/GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2+
)prune_low_magnitude_conv1d_1/GreaterEqualÛ
5prune_low_magnitude_conv1d_1/LessEqual/ReadVariableOpReadVariableOp4prune_low_magnitude_conv1d_1_readvariableop_resource.^prune_low_magnitude_conv1d_1/AssignVariableOpN^prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	27
5prune_low_magnitude_conv1d_1/LessEqual/ReadVariableOpè
(prune_low_magnitude_conv1d_1/LessEqual/yConstN^prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2*
(prune_low_magnitude_conv1d_1/LessEqual/yð
&prune_low_magnitude_conv1d_1/LessEqual	LessEqual=prune_low_magnitude_conv1d_1/LessEqual/ReadVariableOp:value:01prune_low_magnitude_conv1d_1/LessEqual/y:output:0*
T0	*
_output_shapes
: 2(
&prune_low_magnitude_conv1d_1/LessEqualÞ
#prune_low_magnitude_conv1d_1/Less/xConstN^prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2%
#prune_low_magnitude_conv1d_1/Less/xÜ
#prune_low_magnitude_conv1d_1/Less/yConstN^prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2%
#prune_low_magnitude_conv1d_1/Less/yË
!prune_low_magnitude_conv1d_1/LessLess,prune_low_magnitude_conv1d_1/Less/x:output:0,prune_low_magnitude_conv1d_1/Less/y:output:0*
T0	*
_output_shapes
: 2#
!prune_low_magnitude_conv1d_1/LessÈ
&prune_low_magnitude_conv1d_1/LogicalOr	LogicalOr*prune_low_magnitude_conv1d_1/LessEqual:z:0%prune_low_magnitude_conv1d_1/Less:z:0*
_output_shapes
: 2(
&prune_low_magnitude_conv1d_1/LogicalOrÓ
'prune_low_magnitude_conv1d_1/LogicalAnd
LogicalAnd-prune_low_magnitude_conv1d_1/GreaterEqual:z:0*prune_low_magnitude_conv1d_1/LogicalOr:z:0*
_output_shapes
: 2)
'prune_low_magnitude_conv1d_1/LogicalAndÏ
/prune_low_magnitude_conv1d_1/Sub/ReadVariableOpReadVariableOp4prune_low_magnitude_conv1d_1_readvariableop_resource.^prune_low_magnitude_conv1d_1/AssignVariableOpN^prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	21
/prune_low_magnitude_conv1d_1/Sub/ReadVariableOpÚ
"prune_low_magnitude_conv1d_1/Sub/yConstN^prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2$
"prune_low_magnitude_conv1d_1/Sub/yÒ
 prune_low_magnitude_conv1d_1/SubSub7prune_low_magnitude_conv1d_1/Sub/ReadVariableOp:value:0+prune_low_magnitude_conv1d_1/Sub/y:output:0*
T0	*
_output_shapes
: 2"
 prune_low_magnitude_conv1d_1/Subä
'prune_low_magnitude_conv1d_1/FloorMod/yConstN^prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2)
'prune_low_magnitude_conv1d_1/FloorMod/yÓ
%prune_low_magnitude_conv1d_1/FloorModFloorMod$prune_low_magnitude_conv1d_1/Sub:z:00prune_low_magnitude_conv1d_1/FloorMod/y:output:0*
T0	*
_output_shapes
: 2'
%prune_low_magnitude_conv1d_1/FloorModÞ
$prune_low_magnitude_conv1d_1/Equal/yConstN^prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2&
$prune_low_magnitude_conv1d_1/Equal/yÌ
"prune_low_magnitude_conv1d_1/EqualEqual)prune_low_magnitude_conv1d_1/FloorMod:z:0-prune_low_magnitude_conv1d_1/Equal/y:output:0*
T0	*
_output_shapes
: 2$
"prune_low_magnitude_conv1d_1/EqualÑ
)prune_low_magnitude_conv1d_1/LogicalAnd_1
LogicalAnd+prune_low_magnitude_conv1d_1/LogicalAnd:z:0&prune_low_magnitude_conv1d_1/Equal:z:0*
_output_shapes
: 2+
)prune_low_magnitude_conv1d_1/LogicalAnd_1º
!prune_low_magnitude_conv1d_1/condIf-prune_low_magnitude_conv1d_1/LogicalAnd_1:z:04prune_low_magnitude_conv1d_1_readvariableop_resource)prune_low_magnitude_conv1d_1_cond_input_1)prune_low_magnitude_conv1d_1_cond_input_2)prune_low_magnitude_conv1d_1_cond_input_3-prune_low_magnitude_conv1d_1/LogicalAnd_1:z:0.^prune_low_magnitude_conv1d_1/AssignVariableOp*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*A
else_branch2R0
.prune_low_magnitude_conv1d_1_cond_false_317644*
output_shapes
: *@
then_branch1R/
-prune_low_magnitude_conv1d_1_cond_true_3176432#
!prune_low_magnitude_conv1d_1/cond±
*prune_low_magnitude_conv1d_1/cond/IdentityIdentity*prune_low_magnitude_conv1d_1/cond:output:0*
T0
*
_output_shapes
: 2,
*prune_low_magnitude_conv1d_1/cond/Identityé
%prune_low_magnitude_conv1d_1/update_1NoOpN^prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard/Identity+^prune_low_magnitude_conv1d_1/cond/Identity*
_output_shapes
 2'
%prune_low_magnitude_conv1d_1/update_1Ð
/prune_low_magnitude_conv1d_1/Mul/ReadVariableOpReadVariableOp)prune_low_magnitude_conv1d_1_cond_input_1*"
_output_shapes
:*
dtype021
/prune_low_magnitude_conv1d_1/Mul/ReadVariableOpø
1prune_low_magnitude_conv1d_1/Mul/ReadVariableOp_1ReadVariableOp)prune_low_magnitude_conv1d_1_cond_input_2"^prune_low_magnitude_conv1d_1/cond*"
_output_shapes
:*
dtype023
1prune_low_magnitude_conv1d_1/Mul/ReadVariableOp_1ì
 prune_low_magnitude_conv1d_1/MulMul7prune_low_magnitude_conv1d_1/Mul/ReadVariableOp:value:09prune_low_magnitude_conv1d_1/Mul/ReadVariableOp_1:value:0*
T0*"
_output_shapes
:2"
 prune_low_magnitude_conv1d_1/MulÀ
/prune_low_magnitude_conv1d_1/AssignVariableOp_1AssignVariableOp)prune_low_magnitude_conv1d_1_cond_input_1$prune_low_magnitude_conv1d_1/Mul:z:00^prune_low_magnitude_conv1d_1/Mul/ReadVariableOp"^prune_low_magnitude_conv1d_1/cond*
_output_shapes
 *
dtype021
/prune_low_magnitude_conv1d_1/AssignVariableOp_1Ð
'prune_low_magnitude_conv1d_1/group_depsNoOp0^prune_low_magnitude_conv1d_1/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2)
'prune_low_magnitude_conv1d_1/group_depsÌ
)prune_low_magnitude_conv1d_1/group_deps_1NoOp(^prune_low_magnitude_conv1d_1/group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2+
)prune_low_magnitude_conv1d_1/group_deps_1ª
2prune_low_magnitude_conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2prune_low_magnitude_conv1d_1/conv1d/ExpandDims/dim
.prune_low_magnitude_conv1d_1/conv1d/ExpandDims
ExpandDims-prune_low_magnitude_conv1d/Relu:activations:0;prune_low_magnitude_conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	20
.prune_low_magnitude_conv1d_1/conv1d/ExpandDims¢
?prune_low_magnitude_conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp)prune_low_magnitude_conv1d_1_cond_input_10^prune_low_magnitude_conv1d_1/AssignVariableOp_1*"
_output_shapes
:*
dtype02A
?prune_low_magnitude_conv1d_1/conv1d/ExpandDims_1/ReadVariableOp®
4prune_low_magnitude_conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4prune_low_magnitude_conv1d_1/conv1d/ExpandDims_1/dim«
0prune_low_magnitude_conv1d_1/conv1d/ExpandDims_1
ExpandDimsGprune_low_magnitude_conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0=prune_low_magnitude_conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0prune_low_magnitude_conv1d_1/conv1d/ExpandDims_1¬
#prune_low_magnitude_conv1d_1/conv1dConv2D7prune_low_magnitude_conv1d_1/conv1d/ExpandDims:output:09prune_low_magnitude_conv1d_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	*
paddingVALID*
strides
2%
#prune_low_magnitude_conv1d_1/conv1dá
+prune_low_magnitude_conv1d_1/conv1d/SqueezeSqueeze,prune_low_magnitude_conv1d_1/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	*
squeeze_dims
2-
+prune_low_magnitude_conv1d_1/conv1d/Squeezeã
3prune_low_magnitude_conv1d_1/BiasAdd/ReadVariableOpReadVariableOp<prune_low_magnitude_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3prune_low_magnitude_conv1d_1/BiasAdd/ReadVariableOp
$prune_low_magnitude_conv1d_1/BiasAddBiasAdd4prune_low_magnitude_conv1d_1/conv1d/Squeeze:output:0;prune_low_magnitude_conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	2&
$prune_low_magnitude_conv1d_1/BiasAdd´
!prune_low_magnitude_conv1d_1/ReluRelu-prune_low_magnitude_conv1d_1/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	2#
!prune_low_magnitude_conv1d_1/ReluÇ
+prune_low_magnitude_conv1d_2/ReadVariableOpReadVariableOp4prune_low_magnitude_conv1d_2_readvariableop_resource*
_output_shapes
: *
dtype0	2-
+prune_low_magnitude_conv1d_2/ReadVariableOp
"prune_low_magnitude_conv1d_2/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"prune_low_magnitude_conv1d_2/add/yÐ
 prune_low_magnitude_conv1d_2/addAddV23prune_low_magnitude_conv1d_2/ReadVariableOp:value:0+prune_low_magnitude_conv1d_2/add/y:output:0*
T0	*
_output_shapes
: 2"
 prune_low_magnitude_conv1d_2/add
-prune_low_magnitude_conv1d_2/AssignVariableOpAssignVariableOp4prune_low_magnitude_conv1d_2_readvariableop_resource$prune_low_magnitude_conv1d_2/add:z:0,^prune_low_magnitude_conv1d_2/ReadVariableOp*
_output_shapes
 *
dtype0	2/
-prune_low_magnitude_conv1d_2/AssignVariableOp
#prune_low_magnitude_conv1d_2/updateNoOp.^prune_low_magnitude_conv1d_2/AssignVariableOp*
_output_shapes
 2%
#prune_low_magnitude_conv1d_2/update¡
@prune_low_magnitude_conv1d_2/assert_greater_equal/ReadVariableOpReadVariableOp4prune_low_magnitude_conv1d_2_readvariableop_resource.^prune_low_magnitude_conv1d_2/AssignVariableOp*
_output_shapes
: *
dtype0	2B
@prune_low_magnitude_conv1d_2/assert_greater_equal/ReadVariableOp¬
3prune_low_magnitude_conv1d_2/assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R25
3prune_low_magnitude_conv1d_2/assert_greater_equal/y¹
>prune_low_magnitude_conv1d_2/assert_greater_equal/GreaterEqualGreaterEqualHprune_low_magnitude_conv1d_2/assert_greater_equal/ReadVariableOp:value:0<prune_low_magnitude_conv1d_2/assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2@
>prune_low_magnitude_conv1d_2/assert_greater_equal/GreaterEqualµ
7prune_low_magnitude_conv1d_2/assert_greater_equal/ConstConst*
_output_shapes
: *
dtype0*
valueB 29
7prune_low_magnitude_conv1d_2/assert_greater_equal/Const
5prune_low_magnitude_conv1d_2/assert_greater_equal/AllAllBprune_low_magnitude_conv1d_2/assert_greater_equal/GreaterEqual:z:0@prune_low_magnitude_conv1d_2/assert_greater_equal/Const:output:0*
_output_shapes
: 27
5prune_low_magnitude_conv1d_2/assert_greater_equal/AllÐ
>prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2@
>prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/Constð
@prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2B
@prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/Const_1
@prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*Z
valueQBO BIx (prune_low_magnitude_conv1d_2/assert_greater_equal/ReadVariableOp:0) = 2B
@prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/Const_2
@prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*M
valueDBB B<y (prune_low_magnitude_conv1d_2/assert_greater_equal/y:0) = 2B
@prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/Const_3Ê
Dprune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuardIf>prune_low_magnitude_conv1d_2/assert_greater_equal/All:output:0>prune_low_magnitude_conv1d_2/assert_greater_equal/All:output:0Hprune_low_magnitude_conv1d_2/assert_greater_equal/ReadVariableOp:value:0<prune_low_magnitude_conv1d_2/assert_greater_equal/y:output:0E^prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *d
else_branchURS
Qprune_low_magnitude_conv1d_2_assert_greater_equal_Assert_AssertGuard_false_317753*
output_shapes
: *c
then_branchTRR
Pprune_low_magnitude_conv1d_2_assert_greater_equal_Assert_AssertGuard_true_3177522F
Dprune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard
Mprune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard/IdentityIdentityMprune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 2O
Mprune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard/Identity
Mprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOp4prune_low_magnitude_conv1d_2_readvariableop_resource.^prune_low_magnitude_conv1d_2/AssignVariableOpN^prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2O
Mprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/ReadVariableOp
Dprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/sub/yConstN^prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2F
Dprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/sub/yÖ
Bprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/subSubUprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/ReadVariableOp:value:0Mprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2D
Bprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/sub
Cprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/CastCastFprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2E
Cprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/Cast©
Hprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/truediv/yConstN^prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 * ¨G2J
Hprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/truediv/yØ
Fprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/truedivRealDivGprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/Cast:y:0Qprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2H
Fprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/truediv©
Hprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/Maximum/xConstN^prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *    2J
Hprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/Maximum/xÛ
Fprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/MaximumMaximumQprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/Maximum/x:output:0Jprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2H
Fprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/Maximum©
Hprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/Minimum/xConstN^prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2J
Hprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/Minimum/xÛ
Fprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/MinimumMinimumQprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/Minimum/x:output:0Jprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2H
Fprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/Minimum¥
Fprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/sub_1/xConstN^prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2H
Fprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/sub_1/xÑ
Dprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/sub_1SubOprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/sub_1/x:output:0Jprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2F
Dprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/sub_1¡
Dprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/Pow/yConstN^prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  @@2F
Dprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/Pow/yÉ
Bprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/PowPowHprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/sub_1:z:0Mprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2D
Bprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/Pow¡
Dprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/Mul/xConstN^prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *¾2F
Dprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/Mul/xÇ
Bprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/MulMulMprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/Mul/x:output:0Fprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2D
Bprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/Mul«
Iprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/sparsity/yConstN^prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2K
Iprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/sparsity/yÖ
Gprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/sparsityAddFprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/Mul:z:0Rprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2I
Gprune_low_magnitude_conv1d_2/polynomial_decay_pruning_schedule/sparsityá
8prune_low_magnitude_conv1d_2/GreaterEqual/ReadVariableOpReadVariableOp4prune_low_magnitude_conv1d_2_readvariableop_resource.^prune_low_magnitude_conv1d_2/AssignVariableOpN^prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2:
8prune_low_magnitude_conv1d_2/GreaterEqual/ReadVariableOpì
+prune_low_magnitude_conv1d_2/GreaterEqual/yConstN^prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2-
+prune_low_magnitude_conv1d_2/GreaterEqual/yÿ
)prune_low_magnitude_conv1d_2/GreaterEqualGreaterEqual@prune_low_magnitude_conv1d_2/GreaterEqual/ReadVariableOp:value:04prune_low_magnitude_conv1d_2/GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2+
)prune_low_magnitude_conv1d_2/GreaterEqualÛ
5prune_low_magnitude_conv1d_2/LessEqual/ReadVariableOpReadVariableOp4prune_low_magnitude_conv1d_2_readvariableop_resource.^prune_low_magnitude_conv1d_2/AssignVariableOpN^prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	27
5prune_low_magnitude_conv1d_2/LessEqual/ReadVariableOpè
(prune_low_magnitude_conv1d_2/LessEqual/yConstN^prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2*
(prune_low_magnitude_conv1d_2/LessEqual/yð
&prune_low_magnitude_conv1d_2/LessEqual	LessEqual=prune_low_magnitude_conv1d_2/LessEqual/ReadVariableOp:value:01prune_low_magnitude_conv1d_2/LessEqual/y:output:0*
T0	*
_output_shapes
: 2(
&prune_low_magnitude_conv1d_2/LessEqualÞ
#prune_low_magnitude_conv1d_2/Less/xConstN^prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2%
#prune_low_magnitude_conv1d_2/Less/xÜ
#prune_low_magnitude_conv1d_2/Less/yConstN^prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2%
#prune_low_magnitude_conv1d_2/Less/yË
!prune_low_magnitude_conv1d_2/LessLess,prune_low_magnitude_conv1d_2/Less/x:output:0,prune_low_magnitude_conv1d_2/Less/y:output:0*
T0	*
_output_shapes
: 2#
!prune_low_magnitude_conv1d_2/LessÈ
&prune_low_magnitude_conv1d_2/LogicalOr	LogicalOr*prune_low_magnitude_conv1d_2/LessEqual:z:0%prune_low_magnitude_conv1d_2/Less:z:0*
_output_shapes
: 2(
&prune_low_magnitude_conv1d_2/LogicalOrÓ
'prune_low_magnitude_conv1d_2/LogicalAnd
LogicalAnd-prune_low_magnitude_conv1d_2/GreaterEqual:z:0*prune_low_magnitude_conv1d_2/LogicalOr:z:0*
_output_shapes
: 2)
'prune_low_magnitude_conv1d_2/LogicalAndÏ
/prune_low_magnitude_conv1d_2/Sub/ReadVariableOpReadVariableOp4prune_low_magnitude_conv1d_2_readvariableop_resource.^prune_low_magnitude_conv1d_2/AssignVariableOpN^prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	21
/prune_low_magnitude_conv1d_2/Sub/ReadVariableOpÚ
"prune_low_magnitude_conv1d_2/Sub/yConstN^prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2$
"prune_low_magnitude_conv1d_2/Sub/yÒ
 prune_low_magnitude_conv1d_2/SubSub7prune_low_magnitude_conv1d_2/Sub/ReadVariableOp:value:0+prune_low_magnitude_conv1d_2/Sub/y:output:0*
T0	*
_output_shapes
: 2"
 prune_low_magnitude_conv1d_2/Subä
'prune_low_magnitude_conv1d_2/FloorMod/yConstN^prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2)
'prune_low_magnitude_conv1d_2/FloorMod/yÓ
%prune_low_magnitude_conv1d_2/FloorModFloorMod$prune_low_magnitude_conv1d_2/Sub:z:00prune_low_magnitude_conv1d_2/FloorMod/y:output:0*
T0	*
_output_shapes
: 2'
%prune_low_magnitude_conv1d_2/FloorModÞ
$prune_low_magnitude_conv1d_2/Equal/yConstN^prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2&
$prune_low_magnitude_conv1d_2/Equal/yÌ
"prune_low_magnitude_conv1d_2/EqualEqual)prune_low_magnitude_conv1d_2/FloorMod:z:0-prune_low_magnitude_conv1d_2/Equal/y:output:0*
T0	*
_output_shapes
: 2$
"prune_low_magnitude_conv1d_2/EqualÑ
)prune_low_magnitude_conv1d_2/LogicalAnd_1
LogicalAnd+prune_low_magnitude_conv1d_2/LogicalAnd:z:0&prune_low_magnitude_conv1d_2/Equal:z:0*
_output_shapes
: 2+
)prune_low_magnitude_conv1d_2/LogicalAnd_1º
!prune_low_magnitude_conv1d_2/condIf-prune_low_magnitude_conv1d_2/LogicalAnd_1:z:04prune_low_magnitude_conv1d_2_readvariableop_resource)prune_low_magnitude_conv1d_2_cond_input_1)prune_low_magnitude_conv1d_2_cond_input_2)prune_low_magnitude_conv1d_2_cond_input_3-prune_low_magnitude_conv1d_2/LogicalAnd_1:z:0.^prune_low_magnitude_conv1d_2/AssignVariableOp*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*A
else_branch2R0
.prune_low_magnitude_conv1d_2_cond_false_317810*
output_shapes
: *@
then_branch1R/
-prune_low_magnitude_conv1d_2_cond_true_3178092#
!prune_low_magnitude_conv1d_2/cond±
*prune_low_magnitude_conv1d_2/cond/IdentityIdentity*prune_low_magnitude_conv1d_2/cond:output:0*
T0
*
_output_shapes
: 2,
*prune_low_magnitude_conv1d_2/cond/Identityé
%prune_low_magnitude_conv1d_2/update_1NoOpN^prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard/Identity+^prune_low_magnitude_conv1d_2/cond/Identity*
_output_shapes
 2'
%prune_low_magnitude_conv1d_2/update_1Ð
/prune_low_magnitude_conv1d_2/Mul/ReadVariableOpReadVariableOp)prune_low_magnitude_conv1d_2_cond_input_1*"
_output_shapes
:*
dtype021
/prune_low_magnitude_conv1d_2/Mul/ReadVariableOpø
1prune_low_magnitude_conv1d_2/Mul/ReadVariableOp_1ReadVariableOp)prune_low_magnitude_conv1d_2_cond_input_2"^prune_low_magnitude_conv1d_2/cond*"
_output_shapes
:*
dtype023
1prune_low_magnitude_conv1d_2/Mul/ReadVariableOp_1ì
 prune_low_magnitude_conv1d_2/MulMul7prune_low_magnitude_conv1d_2/Mul/ReadVariableOp:value:09prune_low_magnitude_conv1d_2/Mul/ReadVariableOp_1:value:0*
T0*"
_output_shapes
:2"
 prune_low_magnitude_conv1d_2/MulÀ
/prune_low_magnitude_conv1d_2/AssignVariableOp_1AssignVariableOp)prune_low_magnitude_conv1d_2_cond_input_1$prune_low_magnitude_conv1d_2/Mul:z:00^prune_low_magnitude_conv1d_2/Mul/ReadVariableOp"^prune_low_magnitude_conv1d_2/cond*
_output_shapes
 *
dtype021
/prune_low_magnitude_conv1d_2/AssignVariableOp_1Ð
'prune_low_magnitude_conv1d_2/group_depsNoOp0^prune_low_magnitude_conv1d_2/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2)
'prune_low_magnitude_conv1d_2/group_depsÌ
)prune_low_magnitude_conv1d_2/group_deps_1NoOp(^prune_low_magnitude_conv1d_2/group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2+
)prune_low_magnitude_conv1d_2/group_deps_1ª
2prune_low_magnitude_conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2prune_low_magnitude_conv1d_2/conv1d/ExpandDims/dim
.prune_low_magnitude_conv1d_2/conv1d/ExpandDims
ExpandDims/prune_low_magnitude_conv1d_1/Relu:activations:0;prune_low_magnitude_conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	20
.prune_low_magnitude_conv1d_2/conv1d/ExpandDims¢
?prune_low_magnitude_conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp)prune_low_magnitude_conv1d_2_cond_input_10^prune_low_magnitude_conv1d_2/AssignVariableOp_1*"
_output_shapes
:*
dtype02A
?prune_low_magnitude_conv1d_2/conv1d/ExpandDims_1/ReadVariableOp®
4prune_low_magnitude_conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4prune_low_magnitude_conv1d_2/conv1d/ExpandDims_1/dim«
0prune_low_magnitude_conv1d_2/conv1d/ExpandDims_1
ExpandDimsGprune_low_magnitude_conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0=prune_low_magnitude_conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0prune_low_magnitude_conv1d_2/conv1d/ExpandDims_1¬
#prune_low_magnitude_conv1d_2/conv1dConv2D7prune_low_magnitude_conv1d_2/conv1d/ExpandDims:output:09prune_low_magnitude_conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	*
paddingVALID*
strides
2%
#prune_low_magnitude_conv1d_2/conv1dá
+prune_low_magnitude_conv1d_2/conv1d/SqueezeSqueeze,prune_low_magnitude_conv1d_2/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	*
squeeze_dims
2-
+prune_low_magnitude_conv1d_2/conv1d/Squeezeã
3prune_low_magnitude_conv1d_2/BiasAdd/ReadVariableOpReadVariableOp<prune_low_magnitude_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3prune_low_magnitude_conv1d_2/BiasAdd/ReadVariableOp
$prune_low_magnitude_conv1d_2/BiasAddBiasAdd4prune_low_magnitude_conv1d_2/conv1d/Squeeze:output:0;prune_low_magnitude_conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2&
$prune_low_magnitude_conv1d_2/BiasAdd´
!prune_low_magnitude_conv1d_2/ReluRelu-prune_low_magnitude_conv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2#
!prune_low_magnitude_conv1d_2/ReluÄ
*prune_low_magnitude_dropout/ReadVariableOpReadVariableOp3prune_low_magnitude_dropout_readvariableop_resource*
_output_shapes
: *
dtype0	2,
*prune_low_magnitude_dropout/ReadVariableOp
!prune_low_magnitude_dropout/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2#
!prune_low_magnitude_dropout/add/yÌ
prune_low_magnitude_dropout/addAddV22prune_low_magnitude_dropout/ReadVariableOp:value:0*prune_low_magnitude_dropout/add/y:output:0*
T0	*
_output_shapes
: 2!
prune_low_magnitude_dropout/add
,prune_low_magnitude_dropout/AssignVariableOpAssignVariableOp3prune_low_magnitude_dropout_readvariableop_resource#prune_low_magnitude_dropout/add:z:0+^prune_low_magnitude_dropout/ReadVariableOp*
_output_shapes
 *
dtype0	2.
,prune_low_magnitude_dropout/AssignVariableOp
"prune_low_magnitude_dropout/updateNoOp-^prune_low_magnitude_dropout/AssignVariableOp*
_output_shapes
 2$
"prune_low_magnitude_dropout/update
?prune_low_magnitude_dropout/assert_greater_equal/ReadVariableOpReadVariableOp3prune_low_magnitude_dropout_readvariableop_resource-^prune_low_magnitude_dropout/AssignVariableOp*
_output_shapes
: *
dtype0	2A
?prune_low_magnitude_dropout/assert_greater_equal/ReadVariableOpª
2prune_low_magnitude_dropout/assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R24
2prune_low_magnitude_dropout/assert_greater_equal/yµ
=prune_low_magnitude_dropout/assert_greater_equal/GreaterEqualGreaterEqualGprune_low_magnitude_dropout/assert_greater_equal/ReadVariableOp:value:0;prune_low_magnitude_dropout/assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2?
=prune_low_magnitude_dropout/assert_greater_equal/GreaterEqual³
6prune_low_magnitude_dropout/assert_greater_equal/ConstConst*
_output_shapes
: *
dtype0*
valueB 28
6prune_low_magnitude_dropout/assert_greater_equal/Const
4prune_low_magnitude_dropout/assert_greater_equal/AllAllAprune_low_magnitude_dropout/assert_greater_equal/GreaterEqual:z:0?prune_low_magnitude_dropout/assert_greater_equal/Const:output:0*
_output_shapes
: 26
4prune_low_magnitude_dropout/assert_greater_equal/AllÎ
=prune_low_magnitude_dropout/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2?
=prune_low_magnitude_dropout/assert_greater_equal/Assert/Constî
?prune_low_magnitude_dropout/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2A
?prune_low_magnitude_dropout/assert_greater_equal/Assert/Const_1
?prune_low_magnitude_dropout/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*Y
valuePBN BHx (prune_low_magnitude_dropout/assert_greater_equal/ReadVariableOp:0) = 2A
?prune_low_magnitude_dropout/assert_greater_equal/Assert/Const_2þ
?prune_low_magnitude_dropout/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*L
valueCBA B;y (prune_low_magnitude_dropout/assert_greater_equal/y:0) = 2A
?prune_low_magnitude_dropout/assert_greater_equal/Assert/Const_3Â
Cprune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuardIf=prune_low_magnitude_dropout/assert_greater_equal/All:output:0=prune_low_magnitude_dropout/assert_greater_equal/All:output:0Gprune_low_magnitude_dropout/assert_greater_equal/ReadVariableOp:value:0;prune_low_magnitude_dropout/assert_greater_equal/y:output:0E^prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *c
else_branchTRR
Pprune_low_magnitude_dropout_assert_greater_equal_Assert_AssertGuard_false_317919*
output_shapes
: *b
then_branchSRQ
Oprune_low_magnitude_dropout_assert_greater_equal_Assert_AssertGuard_true_3179182E
Cprune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard
Lprune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard/IdentityIdentityLprune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 2N
Lprune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard/Identity
Lprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOp3prune_low_magnitude_dropout_readvariableop_resource-^prune_low_magnitude_dropout/AssignVariableOpM^prune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2N
Lprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/ReadVariableOp
Cprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/sub/yConstM^prune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2E
Cprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/sub/yÒ
Aprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/subSubTprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/ReadVariableOp:value:0Lprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2C
Aprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/sub
Bprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/CastCastEprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2D
Bprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/Cast¦
Gprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/truediv/yConstM^prune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 * ¨G2I
Gprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/truediv/yÔ
Eprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/truedivRealDivFprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/Cast:y:0Pprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2G
Eprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/truediv¦
Gprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/Maximum/xConstM^prune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *    2I
Gprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/Maximum/x×
Eprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/MaximumMaximumPprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/Maximum/x:output:0Iprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2G
Eprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/Maximum¦
Gprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/Minimum/xConstM^prune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2I
Gprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/Minimum/x×
Eprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/MinimumMinimumPprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/Minimum/x:output:0Iprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2G
Eprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/Minimum¢
Eprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/sub_1/xConstM^prune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2G
Eprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/sub_1/xÍ
Cprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/sub_1SubNprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/sub_1/x:output:0Iprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2E
Cprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/sub_1
Cprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/Pow/yConstM^prune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  @@2E
Cprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/Pow/yÅ
Aprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/PowPowGprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/sub_1:z:0Lprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2C
Aprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/Pow
Cprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/Mul/xConstM^prune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *¾2E
Cprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/Mul/xÃ
Aprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/MulMulLprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/Mul/x:output:0Eprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2C
Aprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/Mul¨
Hprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/sparsity/yConstM^prune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2J
Hprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/sparsity/yÒ
Fprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/sparsityAddEprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/Mul:z:0Qprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2H
Fprune_low_magnitude_dropout/polynomial_decay_pruning_schedule/sparsityÜ
7prune_low_magnitude_dropout/GreaterEqual/ReadVariableOpReadVariableOp3prune_low_magnitude_dropout_readvariableop_resource-^prune_low_magnitude_dropout/AssignVariableOpM^prune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	29
7prune_low_magnitude_dropout/GreaterEqual/ReadVariableOpé
*prune_low_magnitude_dropout/GreaterEqual/yConstM^prune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2,
*prune_low_magnitude_dropout/GreaterEqual/yû
(prune_low_magnitude_dropout/GreaterEqualGreaterEqual?prune_low_magnitude_dropout/GreaterEqual/ReadVariableOp:value:03prune_low_magnitude_dropout/GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2*
(prune_low_magnitude_dropout/GreaterEqualÖ
4prune_low_magnitude_dropout/LessEqual/ReadVariableOpReadVariableOp3prune_low_magnitude_dropout_readvariableop_resource-^prune_low_magnitude_dropout/AssignVariableOpM^prune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	26
4prune_low_magnitude_dropout/LessEqual/ReadVariableOpå
'prune_low_magnitude_dropout/LessEqual/yConstM^prune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2)
'prune_low_magnitude_dropout/LessEqual/yì
%prune_low_magnitude_dropout/LessEqual	LessEqual<prune_low_magnitude_dropout/LessEqual/ReadVariableOp:value:00prune_low_magnitude_dropout/LessEqual/y:output:0*
T0	*
_output_shapes
: 2'
%prune_low_magnitude_dropout/LessEqualÛ
"prune_low_magnitude_dropout/Less/xConstM^prune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2$
"prune_low_magnitude_dropout/Less/xÙ
"prune_low_magnitude_dropout/Less/yConstM^prune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2$
"prune_low_magnitude_dropout/Less/yÇ
 prune_low_magnitude_dropout/LessLess+prune_low_magnitude_dropout/Less/x:output:0+prune_low_magnitude_dropout/Less/y:output:0*
T0	*
_output_shapes
: 2"
 prune_low_magnitude_dropout/LessÄ
%prune_low_magnitude_dropout/LogicalOr	LogicalOr)prune_low_magnitude_dropout/LessEqual:z:0$prune_low_magnitude_dropout/Less:z:0*
_output_shapes
: 2'
%prune_low_magnitude_dropout/LogicalOrÏ
&prune_low_magnitude_dropout/LogicalAnd
LogicalAnd,prune_low_magnitude_dropout/GreaterEqual:z:0)prune_low_magnitude_dropout/LogicalOr:z:0*
_output_shapes
: 2(
&prune_low_magnitude_dropout/LogicalAndÊ
.prune_low_magnitude_dropout/Sub/ReadVariableOpReadVariableOp3prune_low_magnitude_dropout_readvariableop_resource-^prune_low_magnitude_dropout/AssignVariableOpM^prune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	20
.prune_low_magnitude_dropout/Sub/ReadVariableOp×
!prune_low_magnitude_dropout/Sub/yConstM^prune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2#
!prune_low_magnitude_dropout/Sub/yÎ
prune_low_magnitude_dropout/SubSub6prune_low_magnitude_dropout/Sub/ReadVariableOp:value:0*prune_low_magnitude_dropout/Sub/y:output:0*
T0	*
_output_shapes
: 2!
prune_low_magnitude_dropout/Subá
&prune_low_magnitude_dropout/FloorMod/yConstM^prune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2(
&prune_low_magnitude_dropout/FloorMod/yÏ
$prune_low_magnitude_dropout/FloorModFloorMod#prune_low_magnitude_dropout/Sub:z:0/prune_low_magnitude_dropout/FloorMod/y:output:0*
T0	*
_output_shapes
: 2&
$prune_low_magnitude_dropout/FloorModÛ
#prune_low_magnitude_dropout/Equal/yConstM^prune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2%
#prune_low_magnitude_dropout/Equal/yÈ
!prune_low_magnitude_dropout/EqualEqual(prune_low_magnitude_dropout/FloorMod:z:0,prune_low_magnitude_dropout/Equal/y:output:0*
T0	*
_output_shapes
: 2#
!prune_low_magnitude_dropout/EqualÍ
(prune_low_magnitude_dropout/LogicalAnd_1
LogicalAnd*prune_low_magnitude_dropout/LogicalAnd:z:0%prune_low_magnitude_dropout/Equal:z:0*
_output_shapes
: 2*
(prune_low_magnitude_dropout/LogicalAnd_1Î
 prune_low_magnitude_dropout/condStatelessIf,prune_low_magnitude_dropout/LogicalAnd_1:z:0,prune_low_magnitude_dropout/LogicalAnd_1:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *@
else_branch1R/
-prune_low_magnitude_dropout_cond_false_317976*
output_shapes
: *?
then_branch0R.
,prune_low_magnitude_dropout_cond_true_3179752"
 prune_low_magnitude_dropout/cond®
)prune_low_magnitude_dropout/cond/IdentityIdentity)prune_low_magnitude_dropout/cond:output:0*
T0
*
_output_shapes
: 2+
)prune_low_magnitude_dropout/cond/Identityå
$prune_low_magnitude_dropout/update_1NoOpM^prune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard/Identity*^prune_low_magnitude_dropout/cond/Identity*
_output_shapes
 2&
$prune_low_magnitude_dropout/update_1n
&prune_low_magnitude_dropout/group_depsNoOp*
_output_shapes
 2(
&prune_low_magnitude_dropout/group_deps
)prune_low_magnitude_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?2+
)prune_low_magnitude_dropout/dropout/Constõ
'prune_low_magnitude_dropout/dropout/MulMul/prune_low_magnitude_conv1d_2/Relu:activations:02prune_low_magnitude_dropout/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2)
'prune_low_magnitude_dropout/dropout/Mulµ
)prune_low_magnitude_dropout/dropout/ShapeShape/prune_low_magnitude_conv1d_2/Relu:activations:0*
T0*
_output_shapes
:2+
)prune_low_magnitude_dropout/dropout/Shape
@prune_low_magnitude_dropout/dropout/random_uniform/RandomUniformRandomUniform2prune_low_magnitude_dropout/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	*
dtype02B
@prune_low_magnitude_dropout/dropout/random_uniform/RandomUniform­
2prune_low_magnitude_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >24
2prune_low_magnitude_dropout/dropout/GreaterEqual/y³
0prune_low_magnitude_dropout/dropout/GreaterEqualGreaterEqualIprune_low_magnitude_dropout/dropout/random_uniform/RandomUniform:output:0;prune_low_magnitude_dropout/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	22
0prune_low_magnitude_dropout/dropout/GreaterEqualØ
(prune_low_magnitude_dropout/dropout/CastCast4prune_low_magnitude_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2*
(prune_low_magnitude_dropout/dropout/Castï
)prune_low_magnitude_dropout/dropout/Mul_1Mul+prune_low_magnitude_dropout/dropout/Mul:z:0,prune_low_magnitude_dropout/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2+
)prune_low_magnitude_dropout/dropout/Mul_1Ä
*prune_low_magnitude_flatten/ReadVariableOpReadVariableOp3prune_low_magnitude_flatten_readvariableop_resource*
_output_shapes
: *
dtype0	2,
*prune_low_magnitude_flatten/ReadVariableOp
!prune_low_magnitude_flatten/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2#
!prune_low_magnitude_flatten/add/yÌ
prune_low_magnitude_flatten/addAddV22prune_low_magnitude_flatten/ReadVariableOp:value:0*prune_low_magnitude_flatten/add/y:output:0*
T0	*
_output_shapes
: 2!
prune_low_magnitude_flatten/add
,prune_low_magnitude_flatten/AssignVariableOpAssignVariableOp3prune_low_magnitude_flatten_readvariableop_resource#prune_low_magnitude_flatten/add:z:0+^prune_low_magnitude_flatten/ReadVariableOp*
_output_shapes
 *
dtype0	2.
,prune_low_magnitude_flatten/AssignVariableOp
"prune_low_magnitude_flatten/updateNoOp-^prune_low_magnitude_flatten/AssignVariableOp*
_output_shapes
 2$
"prune_low_magnitude_flatten/update
?prune_low_magnitude_flatten/assert_greater_equal/ReadVariableOpReadVariableOp3prune_low_magnitude_flatten_readvariableop_resource-^prune_low_magnitude_flatten/AssignVariableOp*
_output_shapes
: *
dtype0	2A
?prune_low_magnitude_flatten/assert_greater_equal/ReadVariableOpª
2prune_low_magnitude_flatten/assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R24
2prune_low_magnitude_flatten/assert_greater_equal/yµ
=prune_low_magnitude_flatten/assert_greater_equal/GreaterEqualGreaterEqualGprune_low_magnitude_flatten/assert_greater_equal/ReadVariableOp:value:0;prune_low_magnitude_flatten/assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2?
=prune_low_magnitude_flatten/assert_greater_equal/GreaterEqual³
6prune_low_magnitude_flatten/assert_greater_equal/ConstConst*
_output_shapes
: *
dtype0*
valueB 28
6prune_low_magnitude_flatten/assert_greater_equal/Const
4prune_low_magnitude_flatten/assert_greater_equal/AllAllAprune_low_magnitude_flatten/assert_greater_equal/GreaterEqual:z:0?prune_low_magnitude_flatten/assert_greater_equal/Const:output:0*
_output_shapes
: 26
4prune_low_magnitude_flatten/assert_greater_equal/AllÎ
=prune_low_magnitude_flatten/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2?
=prune_low_magnitude_flatten/assert_greater_equal/Assert/Constî
?prune_low_magnitude_flatten/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2A
?prune_low_magnitude_flatten/assert_greater_equal/Assert/Const_1
?prune_low_magnitude_flatten/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*Y
valuePBN BHx (prune_low_magnitude_flatten/assert_greater_equal/ReadVariableOp:0) = 2A
?prune_low_magnitude_flatten/assert_greater_equal/Assert/Const_2þ
?prune_low_magnitude_flatten/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*L
valueCBA B;y (prune_low_magnitude_flatten/assert_greater_equal/y:0) = 2A
?prune_low_magnitude_flatten/assert_greater_equal/Assert/Const_3Á
Cprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuardIf=prune_low_magnitude_flatten/assert_greater_equal/All:output:0=prune_low_magnitude_flatten/assert_greater_equal/All:output:0Gprune_low_magnitude_flatten/assert_greater_equal/ReadVariableOp:value:0;prune_low_magnitude_flatten/assert_greater_equal/y:output:0D^prune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *c
else_branchTRR
Pprune_low_magnitude_flatten_assert_greater_equal_Assert_AssertGuard_false_318009*
output_shapes
: *b
then_branchSRQ
Oprune_low_magnitude_flatten_assert_greater_equal_Assert_AssertGuard_true_3180082E
Cprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard
Lprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/IdentityIdentityLprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 2N
Lprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity
Lprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOp3prune_low_magnitude_flatten_readvariableop_resource-^prune_low_magnitude_flatten/AssignVariableOpM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2N
Lprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/ReadVariableOp
Cprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/sub/yConstM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2E
Cprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/sub/yÒ
Aprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/subSubTprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/ReadVariableOp:value:0Lprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2C
Aprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/sub
Bprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/CastCastEprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2D
Bprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/Cast¦
Gprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/truediv/yConstM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 * ¨G2I
Gprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/truediv/yÔ
Eprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/truedivRealDivFprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/Cast:y:0Pprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2G
Eprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/truediv¦
Gprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/Maximum/xConstM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *    2I
Gprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/Maximum/x×
Eprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/MaximumMaximumPprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/Maximum/x:output:0Iprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2G
Eprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/Maximum¦
Gprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/Minimum/xConstM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2I
Gprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/Minimum/x×
Eprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/MinimumMinimumPprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/Minimum/x:output:0Iprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2G
Eprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/Minimum¢
Eprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/sub_1/xConstM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2G
Eprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/sub_1/xÍ
Cprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/sub_1SubNprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/sub_1/x:output:0Iprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2E
Cprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/sub_1
Cprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/Pow/yConstM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  @@2E
Cprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/Pow/yÅ
Aprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/PowPowGprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/sub_1:z:0Lprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2C
Aprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/Pow
Cprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/Mul/xConstM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *¾2E
Cprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/Mul/xÃ
Aprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/MulMulLprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/Mul/x:output:0Eprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2C
Aprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/Mul¨
Hprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/sparsity/yConstM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2J
Hprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/sparsity/yÒ
Fprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/sparsityAddEprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/Mul:z:0Qprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2H
Fprune_low_magnitude_flatten/polynomial_decay_pruning_schedule/sparsityÜ
7prune_low_magnitude_flatten/GreaterEqual/ReadVariableOpReadVariableOp3prune_low_magnitude_flatten_readvariableop_resource-^prune_low_magnitude_flatten/AssignVariableOpM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	29
7prune_low_magnitude_flatten/GreaterEqual/ReadVariableOpé
*prune_low_magnitude_flatten/GreaterEqual/yConstM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2,
*prune_low_magnitude_flatten/GreaterEqual/yû
(prune_low_magnitude_flatten/GreaterEqualGreaterEqual?prune_low_magnitude_flatten/GreaterEqual/ReadVariableOp:value:03prune_low_magnitude_flatten/GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2*
(prune_low_magnitude_flatten/GreaterEqualÖ
4prune_low_magnitude_flatten/LessEqual/ReadVariableOpReadVariableOp3prune_low_magnitude_flatten_readvariableop_resource-^prune_low_magnitude_flatten/AssignVariableOpM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	26
4prune_low_magnitude_flatten/LessEqual/ReadVariableOpå
'prune_low_magnitude_flatten/LessEqual/yConstM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2)
'prune_low_magnitude_flatten/LessEqual/yì
%prune_low_magnitude_flatten/LessEqual	LessEqual<prune_low_magnitude_flatten/LessEqual/ReadVariableOp:value:00prune_low_magnitude_flatten/LessEqual/y:output:0*
T0	*
_output_shapes
: 2'
%prune_low_magnitude_flatten/LessEqualÛ
"prune_low_magnitude_flatten/Less/xConstM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2$
"prune_low_magnitude_flatten/Less/xÙ
"prune_low_magnitude_flatten/Less/yConstM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2$
"prune_low_magnitude_flatten/Less/yÇ
 prune_low_magnitude_flatten/LessLess+prune_low_magnitude_flatten/Less/x:output:0+prune_low_magnitude_flatten/Less/y:output:0*
T0	*
_output_shapes
: 2"
 prune_low_magnitude_flatten/LessÄ
%prune_low_magnitude_flatten/LogicalOr	LogicalOr)prune_low_magnitude_flatten/LessEqual:z:0$prune_low_magnitude_flatten/Less:z:0*
_output_shapes
: 2'
%prune_low_magnitude_flatten/LogicalOrÏ
&prune_low_magnitude_flatten/LogicalAnd
LogicalAnd,prune_low_magnitude_flatten/GreaterEqual:z:0)prune_low_magnitude_flatten/LogicalOr:z:0*
_output_shapes
: 2(
&prune_low_magnitude_flatten/LogicalAndÊ
.prune_low_magnitude_flatten/Sub/ReadVariableOpReadVariableOp3prune_low_magnitude_flatten_readvariableop_resource-^prune_low_magnitude_flatten/AssignVariableOpM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	20
.prune_low_magnitude_flatten/Sub/ReadVariableOp×
!prune_low_magnitude_flatten/Sub/yConstM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2#
!prune_low_magnitude_flatten/Sub/yÎ
prune_low_magnitude_flatten/SubSub6prune_low_magnitude_flatten/Sub/ReadVariableOp:value:0*prune_low_magnitude_flatten/Sub/y:output:0*
T0	*
_output_shapes
: 2!
prune_low_magnitude_flatten/Subá
&prune_low_magnitude_flatten/FloorMod/yConstM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2(
&prune_low_magnitude_flatten/FloorMod/yÏ
$prune_low_magnitude_flatten/FloorModFloorMod#prune_low_magnitude_flatten/Sub:z:0/prune_low_magnitude_flatten/FloorMod/y:output:0*
T0	*
_output_shapes
: 2&
$prune_low_magnitude_flatten/FloorModÛ
#prune_low_magnitude_flatten/Equal/yConstM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2%
#prune_low_magnitude_flatten/Equal/yÈ
!prune_low_magnitude_flatten/EqualEqual(prune_low_magnitude_flatten/FloorMod:z:0,prune_low_magnitude_flatten/Equal/y:output:0*
T0	*
_output_shapes
: 2#
!prune_low_magnitude_flatten/EqualÍ
(prune_low_magnitude_flatten/LogicalAnd_1
LogicalAnd*prune_low_magnitude_flatten/LogicalAnd:z:0%prune_low_magnitude_flatten/Equal:z:0*
_output_shapes
: 2*
(prune_low_magnitude_flatten/LogicalAnd_1Î
 prune_low_magnitude_flatten/condStatelessIf,prune_low_magnitude_flatten/LogicalAnd_1:z:0,prune_low_magnitude_flatten/LogicalAnd_1:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *@
else_branch1R/
-prune_low_magnitude_flatten_cond_false_318066*
output_shapes
: *?
then_branch0R.
,prune_low_magnitude_flatten_cond_true_3180652"
 prune_low_magnitude_flatten/cond®
)prune_low_magnitude_flatten/cond/IdentityIdentity)prune_low_magnitude_flatten/cond:output:0*
T0
*
_output_shapes
: 2+
)prune_low_magnitude_flatten/cond/Identityå
$prune_low_magnitude_flatten/update_1NoOpM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*^prune_low_magnitude_flatten/cond/Identity*
_output_shapes
 2&
$prune_low_magnitude_flatten/update_1n
&prune_low_magnitude_flatten/group_depsNoOp*
_output_shapes
 2(
&prune_low_magnitude_flatten/group_deps
!prune_low_magnitude_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿM  2#
!prune_low_magnitude_flatten/Constä
#prune_low_magnitude_flatten/ReshapeReshape-prune_low_magnitude_dropout/dropout/Mul_1:z:0*prune_low_magnitude_flatten/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#prune_low_magnitude_flatten/Reshape¾
(prune_low_magnitude_dense/ReadVariableOpReadVariableOp1prune_low_magnitude_dense_readvariableop_resource*
_output_shapes
: *
dtype0	2*
(prune_low_magnitude_dense/ReadVariableOp
prune_low_magnitude_dense/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2!
prune_low_magnitude_dense/add/yÄ
prune_low_magnitude_dense/addAddV20prune_low_magnitude_dense/ReadVariableOp:value:0(prune_low_magnitude_dense/add/y:output:0*
T0	*
_output_shapes
: 2
prune_low_magnitude_dense/add
*prune_low_magnitude_dense/AssignVariableOpAssignVariableOp1prune_low_magnitude_dense_readvariableop_resource!prune_low_magnitude_dense/add:z:0)^prune_low_magnitude_dense/ReadVariableOp*
_output_shapes
 *
dtype0	2,
*prune_low_magnitude_dense/AssignVariableOp
 prune_low_magnitude_dense/updateNoOp+^prune_low_magnitude_dense/AssignVariableOp*
_output_shapes
 2"
 prune_low_magnitude_dense/update
=prune_low_magnitude_dense/assert_greater_equal/ReadVariableOpReadVariableOp1prune_low_magnitude_dense_readvariableop_resource+^prune_low_magnitude_dense/AssignVariableOp*
_output_shapes
: *
dtype0	2?
=prune_low_magnitude_dense/assert_greater_equal/ReadVariableOp¦
0prune_low_magnitude_dense/assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R22
0prune_low_magnitude_dense/assert_greater_equal/y­
;prune_low_magnitude_dense/assert_greater_equal/GreaterEqualGreaterEqualEprune_low_magnitude_dense/assert_greater_equal/ReadVariableOp:value:09prune_low_magnitude_dense/assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2=
;prune_low_magnitude_dense/assert_greater_equal/GreaterEqual¯
4prune_low_magnitude_dense/assert_greater_equal/ConstConst*
_output_shapes
: *
dtype0*
valueB 26
4prune_low_magnitude_dense/assert_greater_equal/Const
2prune_low_magnitude_dense/assert_greater_equal/AllAll?prune_low_magnitude_dense/assert_greater_equal/GreaterEqual:z:0=prune_low_magnitude_dense/assert_greater_equal/Const:output:0*
_output_shapes
: 24
2prune_low_magnitude_dense/assert_greater_equal/AllÊ
;prune_low_magnitude_dense/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2=
;prune_low_magnitude_dense/assert_greater_equal/Assert/Constê
=prune_low_magnitude_dense/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2?
=prune_low_magnitude_dense/assert_greater_equal/Assert/Const_1
=prune_low_magnitude_dense/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (prune_low_magnitude_dense/assert_greater_equal/ReadVariableOp:0) = 2?
=prune_low_magnitude_dense/assert_greater_equal/Assert/Const_2ø
=prune_low_magnitude_dense/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*J
valueAB? B9y (prune_low_magnitude_dense/assert_greater_equal/y:0) = 2?
=prune_low_magnitude_dense/assert_greater_equal/Assert/Const_3±
Aprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuardIf;prune_low_magnitude_dense/assert_greater_equal/All:output:0;prune_low_magnitude_dense/assert_greater_equal/All:output:0Eprune_low_magnitude_dense/assert_greater_equal/ReadVariableOp:value:09prune_low_magnitude_dense/assert_greater_equal/y:output:0D^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *a
else_branchRRP
Nprune_low_magnitude_dense_assert_greater_equal_Assert_AssertGuard_false_318093*
output_shapes
: *`
then_branchQRO
Mprune_low_magnitude_dense_assert_greater_equal_Assert_AssertGuard_true_3180922C
Aprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard
Jprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/IdentityIdentityJprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 2L
Jprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identityü
Jprune_low_magnitude_dense/polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOp1prune_low_magnitude_dense_readvariableop_resource+^prune_low_magnitude_dense/AssignVariableOpK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2L
Jprune_low_magnitude_dense/polynomial_decay_pruning_schedule/ReadVariableOp
Aprune_low_magnitude_dense/polynomial_decay_pruning_schedule/sub/yConstK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2C
Aprune_low_magnitude_dense/polynomial_decay_pruning_schedule/sub/yÊ
?prune_low_magnitude_dense/polynomial_decay_pruning_schedule/subSubRprune_low_magnitude_dense/polynomial_decay_pruning_schedule/ReadVariableOp:value:0Jprune_low_magnitude_dense/polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2A
?prune_low_magnitude_dense/polynomial_decay_pruning_schedule/sub
@prune_low_magnitude_dense/polynomial_decay_pruning_schedule/CastCastCprune_low_magnitude_dense/polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2B
@prune_low_magnitude_dense/polynomial_decay_pruning_schedule/Cast 
Eprune_low_magnitude_dense/polynomial_decay_pruning_schedule/truediv/yConstK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 * ¨G2G
Eprune_low_magnitude_dense/polynomial_decay_pruning_schedule/truediv/yÌ
Cprune_low_magnitude_dense/polynomial_decay_pruning_schedule/truedivRealDivDprune_low_magnitude_dense/polynomial_decay_pruning_schedule/Cast:y:0Nprune_low_magnitude_dense/polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2E
Cprune_low_magnitude_dense/polynomial_decay_pruning_schedule/truediv 
Eprune_low_magnitude_dense/polynomial_decay_pruning_schedule/Maximum/xConstK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *    2G
Eprune_low_magnitude_dense/polynomial_decay_pruning_schedule/Maximum/xÏ
Cprune_low_magnitude_dense/polynomial_decay_pruning_schedule/MaximumMaximumNprune_low_magnitude_dense/polynomial_decay_pruning_schedule/Maximum/x:output:0Gprune_low_magnitude_dense/polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2E
Cprune_low_magnitude_dense/polynomial_decay_pruning_schedule/Maximum 
Eprune_low_magnitude_dense/polynomial_decay_pruning_schedule/Minimum/xConstK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2G
Eprune_low_magnitude_dense/polynomial_decay_pruning_schedule/Minimum/xÏ
Cprune_low_magnitude_dense/polynomial_decay_pruning_schedule/MinimumMinimumNprune_low_magnitude_dense/polynomial_decay_pruning_schedule/Minimum/x:output:0Gprune_low_magnitude_dense/polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2E
Cprune_low_magnitude_dense/polynomial_decay_pruning_schedule/Minimum
Cprune_low_magnitude_dense/polynomial_decay_pruning_schedule/sub_1/xConstK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2E
Cprune_low_magnitude_dense/polynomial_decay_pruning_schedule/sub_1/xÅ
Aprune_low_magnitude_dense/polynomial_decay_pruning_schedule/sub_1SubLprune_low_magnitude_dense/polynomial_decay_pruning_schedule/sub_1/x:output:0Gprune_low_magnitude_dense/polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2C
Aprune_low_magnitude_dense/polynomial_decay_pruning_schedule/sub_1
Aprune_low_magnitude_dense/polynomial_decay_pruning_schedule/Pow/yConstK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  @@2C
Aprune_low_magnitude_dense/polynomial_decay_pruning_schedule/Pow/y½
?prune_low_magnitude_dense/polynomial_decay_pruning_schedule/PowPowEprune_low_magnitude_dense/polynomial_decay_pruning_schedule/sub_1:z:0Jprune_low_magnitude_dense/polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2A
?prune_low_magnitude_dense/polynomial_decay_pruning_schedule/Pow
Aprune_low_magnitude_dense/polynomial_decay_pruning_schedule/Mul/xConstK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *¾2C
Aprune_low_magnitude_dense/polynomial_decay_pruning_schedule/Mul/x»
?prune_low_magnitude_dense/polynomial_decay_pruning_schedule/MulMulJprune_low_magnitude_dense/polynomial_decay_pruning_schedule/Mul/x:output:0Cprune_low_magnitude_dense/polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2A
?prune_low_magnitude_dense/polynomial_decay_pruning_schedule/Mul¢
Fprune_low_magnitude_dense/polynomial_decay_pruning_schedule/sparsity/yConstK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2H
Fprune_low_magnitude_dense/polynomial_decay_pruning_schedule/sparsity/yÊ
Dprune_low_magnitude_dense/polynomial_decay_pruning_schedule/sparsityAddCprune_low_magnitude_dense/polynomial_decay_pruning_schedule/Mul:z:0Oprune_low_magnitude_dense/polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2F
Dprune_low_magnitude_dense/polynomial_decay_pruning_schedule/sparsityÒ
5prune_low_magnitude_dense/GreaterEqual/ReadVariableOpReadVariableOp1prune_low_magnitude_dense_readvariableop_resource+^prune_low_magnitude_dense/AssignVariableOpK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	27
5prune_low_magnitude_dense/GreaterEqual/ReadVariableOpã
(prune_low_magnitude_dense/GreaterEqual/yConstK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2*
(prune_low_magnitude_dense/GreaterEqual/yó
&prune_low_magnitude_dense/GreaterEqualGreaterEqual=prune_low_magnitude_dense/GreaterEqual/ReadVariableOp:value:01prune_low_magnitude_dense/GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2(
&prune_low_magnitude_dense/GreaterEqualÌ
2prune_low_magnitude_dense/LessEqual/ReadVariableOpReadVariableOp1prune_low_magnitude_dense_readvariableop_resource+^prune_low_magnitude_dense/AssignVariableOpK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	24
2prune_low_magnitude_dense/LessEqual/ReadVariableOpß
%prune_low_magnitude_dense/LessEqual/yConstK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2'
%prune_low_magnitude_dense/LessEqual/yä
#prune_low_magnitude_dense/LessEqual	LessEqual:prune_low_magnitude_dense/LessEqual/ReadVariableOp:value:0.prune_low_magnitude_dense/LessEqual/y:output:0*
T0	*
_output_shapes
: 2%
#prune_low_magnitude_dense/LessEqualÕ
 prune_low_magnitude_dense/Less/xConstK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2"
 prune_low_magnitude_dense/Less/xÓ
 prune_low_magnitude_dense/Less/yConstK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2"
 prune_low_magnitude_dense/Less/y¿
prune_low_magnitude_dense/LessLess)prune_low_magnitude_dense/Less/x:output:0)prune_low_magnitude_dense/Less/y:output:0*
T0	*
_output_shapes
: 2 
prune_low_magnitude_dense/Less¼
#prune_low_magnitude_dense/LogicalOr	LogicalOr'prune_low_magnitude_dense/LessEqual:z:0"prune_low_magnitude_dense/Less:z:0*
_output_shapes
: 2%
#prune_low_magnitude_dense/LogicalOrÇ
$prune_low_magnitude_dense/LogicalAnd
LogicalAnd*prune_low_magnitude_dense/GreaterEqual:z:0'prune_low_magnitude_dense/LogicalOr:z:0*
_output_shapes
: 2&
$prune_low_magnitude_dense/LogicalAndÀ
,prune_low_magnitude_dense/Sub/ReadVariableOpReadVariableOp1prune_low_magnitude_dense_readvariableop_resource+^prune_low_magnitude_dense/AssignVariableOpK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2.
,prune_low_magnitude_dense/Sub/ReadVariableOpÑ
prune_low_magnitude_dense/Sub/yConstK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2!
prune_low_magnitude_dense/Sub/yÆ
prune_low_magnitude_dense/SubSub4prune_low_magnitude_dense/Sub/ReadVariableOp:value:0(prune_low_magnitude_dense/Sub/y:output:0*
T0	*
_output_shapes
: 2
prune_low_magnitude_dense/SubÛ
$prune_low_magnitude_dense/FloorMod/yConstK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2&
$prune_low_magnitude_dense/FloorMod/yÇ
"prune_low_magnitude_dense/FloorModFloorMod!prune_low_magnitude_dense/Sub:z:0-prune_low_magnitude_dense/FloorMod/y:output:0*
T0	*
_output_shapes
: 2$
"prune_low_magnitude_dense/FloorModÕ
!prune_low_magnitude_dense/Equal/yConstK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2#
!prune_low_magnitude_dense/Equal/yÀ
prune_low_magnitude_dense/EqualEqual&prune_low_magnitude_dense/FloorMod:z:0*prune_low_magnitude_dense/Equal/y:output:0*
T0	*
_output_shapes
: 2!
prune_low_magnitude_dense/EqualÅ
&prune_low_magnitude_dense/LogicalAnd_1
LogicalAnd(prune_low_magnitude_dense/LogicalAnd:z:0#prune_low_magnitude_dense/Equal:z:0*
_output_shapes
: 2(
&prune_low_magnitude_dense/LogicalAnd_1
prune_low_magnitude_dense/condIf*prune_low_magnitude_dense/LogicalAnd_1:z:01prune_low_magnitude_dense_readvariableop_resource&prune_low_magnitude_dense_cond_input_1&prune_low_magnitude_dense_cond_input_2&prune_low_magnitude_dense_cond_input_3*prune_low_magnitude_dense/LogicalAnd_1:z:0+^prune_low_magnitude_dense/AssignVariableOp*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*>
else_branch/R-
+prune_low_magnitude_dense_cond_false_318150*
output_shapes
: *=
then_branch.R,
*prune_low_magnitude_dense_cond_true_3181492 
prune_low_magnitude_dense/cond¨
'prune_low_magnitude_dense/cond/IdentityIdentity'prune_low_magnitude_dense/cond:output:0*
T0
*
_output_shapes
: 2)
'prune_low_magnitude_dense/cond/IdentityÝ
"prune_low_magnitude_dense/update_1NoOpK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity(^prune_low_magnitude_dense/cond/Identity*
_output_shapes
 2$
"prune_low_magnitude_dense/update_1Å
,prune_low_magnitude_dense/Mul/ReadVariableOpReadVariableOp&prune_low_magnitude_dense_cond_input_1* 
_output_shapes
:
*
dtype02.
,prune_low_magnitude_dense/Mul/ReadVariableOpê
.prune_low_magnitude_dense/Mul/ReadVariableOp_1ReadVariableOp&prune_low_magnitude_dense_cond_input_2^prune_low_magnitude_dense/cond* 
_output_shapes
:
*
dtype020
.prune_low_magnitude_dense/Mul/ReadVariableOp_1Þ
prune_low_magnitude_dense/MulMul4prune_low_magnitude_dense/Mul/ReadVariableOp:value:06prune_low_magnitude_dense/Mul/ReadVariableOp_1:value:0*
T0* 
_output_shapes
:
2
prune_low_magnitude_dense/Mul®
,prune_low_magnitude_dense/AssignVariableOp_1AssignVariableOp&prune_low_magnitude_dense_cond_input_1!prune_low_magnitude_dense/Mul:z:0-^prune_low_magnitude_dense/Mul/ReadVariableOp^prune_low_magnitude_dense/cond*
_output_shapes
 *
dtype02.
,prune_low_magnitude_dense/AssignVariableOp_1Ç
$prune_low_magnitude_dense/group_depsNoOp-^prune_low_magnitude_dense/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2&
$prune_low_magnitude_dense/group_depsÃ
&prune_low_magnitude_dense/group_deps_1NoOp%^prune_low_magnitude_dense/group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2(
&prune_low_magnitude_dense/group_deps_1ú
/prune_low_magnitude_dense/MatMul/ReadVariableOpReadVariableOp&prune_low_magnitude_dense_cond_input_1-^prune_low_magnitude_dense/AssignVariableOp_1* 
_output_shapes
:
*
dtype021
/prune_low_magnitude_dense/MatMul/ReadVariableOpç
 prune_low_magnitude_dense/MatMulMatMul,prune_low_magnitude_flatten/Reshape:output:07prune_low_magnitude_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 prune_low_magnitude_dense/MatMulÚ
0prune_low_magnitude_dense/BiasAdd/ReadVariableOpReadVariableOp9prune_low_magnitude_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0prune_low_magnitude_dense/BiasAdd/ReadVariableOpé
!prune_low_magnitude_dense/BiasAddBiasAdd*prune_low_magnitude_dense/MatMul:product:08prune_low_magnitude_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!prune_low_magnitude_dense/BiasAdd¯
!prune_low_magnitude_dense/SigmoidSigmoid*prune_low_magnitude_dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!prune_low_magnitude_dense/Sigmoid	
IdentityIdentity%prune_low_magnitude_dense/Sigmoid:y:0,^prune_low_magnitude_conv1d/AssignVariableOp.^prune_low_magnitude_conv1d/AssignVariableOp_1C^prune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard ^prune_low_magnitude_conv1d/cond.^prune_low_magnitude_conv1d_1/AssignVariableOp0^prune_low_magnitude_conv1d_1/AssignVariableOp_1E^prune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard"^prune_low_magnitude_conv1d_1/cond.^prune_low_magnitude_conv1d_2/AssignVariableOp0^prune_low_magnitude_conv1d_2/AssignVariableOp_1E^prune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard"^prune_low_magnitude_conv1d_2/cond+^prune_low_magnitude_dense/AssignVariableOp-^prune_low_magnitude_dense/AssignVariableOp_1B^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard^prune_low_magnitude_dense/cond-^prune_low_magnitude_dropout/AssignVariableOpD^prune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard-^prune_low_magnitude_flatten/AssignVariableOpD^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesr
p:ÿÿÿÿÿÿÿÿÿâ	::::::::::::::::::::::2Z
+prune_low_magnitude_conv1d/AssignVariableOp+prune_low_magnitude_conv1d/AssignVariableOp2^
-prune_low_magnitude_conv1d/AssignVariableOp_1-prune_low_magnitude_conv1d/AssignVariableOp_12
Bprune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuardBprune_low_magnitude_conv1d/assert_greater_equal/Assert/AssertGuard2B
prune_low_magnitude_conv1d/condprune_low_magnitude_conv1d/cond2^
-prune_low_magnitude_conv1d_1/AssignVariableOp-prune_low_magnitude_conv1d_1/AssignVariableOp2b
/prune_low_magnitude_conv1d_1/AssignVariableOp_1/prune_low_magnitude_conv1d_1/AssignVariableOp_12
Dprune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuardDprune_low_magnitude_conv1d_1/assert_greater_equal/Assert/AssertGuard2F
!prune_low_magnitude_conv1d_1/cond!prune_low_magnitude_conv1d_1/cond2^
-prune_low_magnitude_conv1d_2/AssignVariableOp-prune_low_magnitude_conv1d_2/AssignVariableOp2b
/prune_low_magnitude_conv1d_2/AssignVariableOp_1/prune_low_magnitude_conv1d_2/AssignVariableOp_12
Dprune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuardDprune_low_magnitude_conv1d_2/assert_greater_equal/Assert/AssertGuard2F
!prune_low_magnitude_conv1d_2/cond!prune_low_magnitude_conv1d_2/cond2X
*prune_low_magnitude_dense/AssignVariableOp*prune_low_magnitude_dense/AssignVariableOp2\
,prune_low_magnitude_dense/AssignVariableOp_1,prune_low_magnitude_dense/AssignVariableOp_12
Aprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuardAprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard2@
prune_low_magnitude_dense/condprune_low_magnitude_dense/cond2\
,prune_low_magnitude_dropout/AssignVariableOp,prune_low_magnitude_dropout/AssignVariableOp2
Cprune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuardCprune_low_magnitude_dropout/assert_greater_equal/Assert/AssertGuard2\
,prune_low_magnitude_flatten/AssignVariableOp,prune_low_magnitude_flatten/AssignVariableOp2
Cprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuardCprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	
 
_user_specified_nameinputs:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ñ
ñ
V__inference_prune_low_magnitude_conv1d_layer_call_and_return_conditional_losses_318575

inputs
mul_readvariableop_resource!
mul_readvariableop_1_resource#
biasadd_readvariableop_resource
identity¢AssignVariableOp4
	no_updateNoOp*
_output_shapes
 2
	no_update8
no_update_1NoOp*
_output_shapes
 2
no_update_1
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*"
_output_shapes
:*
dtype02
Mul/ReadVariableOp
Mul/ReadVariableOp_1ReadVariableOpmul_readvariableop_1_resource*"
_output_shapes
:*
dtype02
Mul/ReadVariableOp_1x
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*"
_output_shapes
:2
Mul
AssignVariableOpAssignVariableOpmul_readvariableop_resourceMul:z:0^Mul/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpw

group_depsNoOp^AssignVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_depsu
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2
group_deps_1p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	2
conv1d/ExpandDims»
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1¸
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	*
squeeze_dims
2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	2
Relu~
IdentityIdentityRelu:activations:0^AssignVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿâ	:::2$
AssignVariableOpAssignVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ÿ
>
cond_false_316791
identity_logicaland_1


identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOp_
IdentityIdentityidentity_logicaland_1^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : 

_output_shapes
: 
³
X
<__inference_prune_low_magnitude_dropout_layer_call_fn_319144

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	* 
_read_only_resource_inputs
 *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*`
f[RY
W__inference_prune_low_magnitude_dropout_layer_call_and_return_conditional_losses_3168162
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÙ	:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	
 
_user_specified_nameinputs
¥
ð
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_317124

inputs
mul_readvariableop_resource!
mul_readvariableop_1_resource#
biasadd_readvariableop_resource
identity¢AssignVariableOp4
	no_updateNoOp*
_output_shapes
 2
	no_update8
no_update_1NoOp*
_output_shapes
 2
no_update_1
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
Mul/ReadVariableOp
Mul/ReadVariableOp_1ReadVariableOpmul_readvariableop_1_resource* 
_output_shapes
:
*
dtype02
Mul/ReadVariableOp_1v
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0* 
_output_shapes
:
2
Mul
AssignVariableOpAssignVariableOpmul_readvariableop_resourceMul:z:0^Mul/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpw

group_depsNoOp^AssignVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_depsu
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2
group_deps_1
MatMul/ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidr
IdentityIdentitySigmoid:y:0^AssignVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ:::2$
AssignVariableOpAssignVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ø

+__inference_sequential_layer_call_fn_318385

inputs
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
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

	*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3173382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿâ	::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	
 
_user_specified_nameinputs:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


3assert_greater_equal_Assert_AssertGuard_true_319266%
!identity_assert_greater_equal_all

placeholder	
placeholder_1	

identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOpk
IdentityIdentity!identity_assert_greater_equal_all^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ö
Ê
Pprune_low_magnitude_conv1d_1_assert_greater_equal_Assert_AssertGuard_true_317586B
>identity_prune_low_magnitude_conv1d_1_assert_greater_equal_all

placeholder	
placeholder_1	

identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOp
IdentityIdentity>identity_prune_low_magnitude_conv1d_1_assert_greater_equal_all^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ã
Ê
4assert_greater_equal_Assert_AssertGuard_false_318618#
assert_assert_greater_equal_all
.
*assert_assert_greater_equal_readvariableop	!
assert_assert_greater_equal_y	

identity_1
¢Assertî
Assert/data_0Const*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2
Assert/data_0
Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2
Assert/data_1
Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2
Assert/data_2~
Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2
Assert/data_4
AssertAssertassert_assert_greater_equal_allAssert/data_0:output:0Assert/data_1:output:0Assert/data_2:output:0*assert_assert_greater_equal_readvariableopAssert/data_4:output:0assert_assert_greater_equal_y*
T

2		*
_output_shapes
 2
Assertk
IdentityIdentityassert_assert_greater_equal_all^Assert*
T0
*
_output_shapes
: 2

Identitya

Identity_1IdentityIdentity:output:0^Assert*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : 2
AssertAssert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¥
ð
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_319430

inputs
mul_readvariableop_resource!
mul_readvariableop_1_resource#
biasadd_readvariableop_resource
identity¢AssignVariableOp4
	no_updateNoOp*
_output_shapes
 2
	no_update8
no_update_1NoOp*
_output_shapes
 2
no_update_1
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
Mul/ReadVariableOp
Mul/ReadVariableOp_1ReadVariableOpmul_readvariableop_1_resource* 
_output_shapes
:
*
dtype02
Mul/ReadVariableOp_1v
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0* 
_output_shapes
:
2
Mul
AssignVariableOpAssignVariableOpmul_readvariableop_resourceMul:z:0^Mul/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpw

group_depsNoOp^AssignVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_depsu
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2
group_deps_1
MatMul/ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidr
IdentityIdentitySigmoid:y:0^AssignVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ:::2$
AssignVariableOpAssignVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ò

cond_false_318891
placeholder
placeholder_1
placeholder_2
placeholder_3
identity_logicaland_1


identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOp_
IdentityIdentityidentity_logicaland_1^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
::::: : 

_output_shapes
: :
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
: 
Ûf
È
X__inference_prune_low_magnitude_conv1d_2_layer_call_and_return_conditional_losses_318987

inputs
readvariableop_resource
cond_input_1
cond_input_2
cond_input_3#
biasadd_readvariableop_resource
identity¢AssignVariableOp¢AssignVariableOp_1¢'assert_greater_equal/Assert/AssertGuard¢condp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOpP
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
add/y\
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: 2
add
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOpA
updateNoOp^AssignVariableOp*
_output_shapes
 2
update­
#assert_greater_equal/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
: *
dtype0	2%
#assert_greater_equal/ReadVariableOpr
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
assert_greater_equal/yÅ
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2#
!assert_greater_equal/GreaterEqual{
assert_greater_equal/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
assert_greater_equal/Const
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/Const:output:0*
_output_shapes
: 2
assert_greater_equal/All
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2#
!assert_greater_equal/Assert/Const¶
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2%
#assert_greater_equal/Assert/Const_1·
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2%
#assert_greater_equal/Assert/Const_2ª
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2%
#assert_greater_equal/Assert/Const_3
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_318834*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_3188332)
'assert_greater_equal/Assert/AssertGuardÃ
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identityú
0polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	22
0polynomial_decay_pruning_schedule/ReadVariableOpÇ
'polynomial_decay_pruning_schedule/sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'polynomial_decay_pruning_schedule/sub/yâ
%polynomial_decay_pruning_schedule/subSub8polynomial_decay_pruning_schedule/ReadVariableOp:value:00polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/sub³
&polynomial_decay_pruning_schedule/CastCast)polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2(
&polynomial_decay_pruning_schedule/CastÒ
+polynomial_decay_pruning_schedule/truediv/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 * ¨G2-
+polynomial_decay_pruning_schedule/truediv/yä
)polynomial_decay_pruning_schedule/truedivRealDiv*polynomial_decay_pruning_schedule/Cast:y:04polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/truedivÒ
+polynomial_decay_pruning_schedule/Maximum/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *    2-
+polynomial_decay_pruning_schedule/Maximum/xç
)polynomial_decay_pruning_schedule/MaximumMaximum4polynomial_decay_pruning_schedule/Maximum/x:output:0-polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/MaximumÒ
+polynomial_decay_pruning_schedule/Minimum/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+polynomial_decay_pruning_schedule/Minimum/xç
)polynomial_decay_pruning_schedule/MinimumMinimum4polynomial_decay_pruning_schedule/Minimum/x:output:0-polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/MinimumÎ
)polynomial_decay_pruning_schedule/sub_1/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)polynomial_decay_pruning_schedule/sub_1/xÝ
'polynomial_decay_pruning_schedule/sub_1Sub2polynomial_decay_pruning_schedule/sub_1/x:output:0-polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2)
'polynomial_decay_pruning_schedule/sub_1Ê
'polynomial_decay_pruning_schedule/Pow/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  @@2)
'polynomial_decay_pruning_schedule/Pow/yÕ
%polynomial_decay_pruning_schedule/PowPow+polynomial_decay_pruning_schedule/sub_1:z:00polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/PowÊ
'polynomial_decay_pruning_schedule/Mul/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *¾2)
'polynomial_decay_pruning_schedule/Mul/xÓ
%polynomial_decay_pruning_schedule/MulMul0polynomial_decay_pruning_schedule/Mul/x:output:0)polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/MulÔ
,polynomial_decay_pruning_schedule/sparsity/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2.
,polynomial_decay_pruning_schedule/sparsity/yâ
*polynomial_decay_pruning_schedule/sparsityAdd)polynomial_decay_pruning_schedule/Mul:z:05polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2,
*polynomial_decay_pruning_schedule/sparsityÐ
GreaterEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOp
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqualÊ
LessEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqual
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
Less/x
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0	*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd¾
Sub/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOp
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
Sub

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorMod
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1û
condIfLogicalAnd_1:z:0readvariableop_resourcecond_input_1cond_input_2cond_input_3LogicalAnd_1:z:0^AssignVariableOp*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*$
else_branchR
cond_false_318891*
output_shapes
: *#
then_branchR
cond_true_3188902
condZ
cond/IdentityIdentitycond:output:0*
T0
*
_output_shapes
: 2
cond/Identityu
update_1NoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 2

update_1y
Mul/ReadVariableOpReadVariableOpcond_input_1*"
_output_shapes
:*
dtype02
Mul/ReadVariableOp
Mul/ReadVariableOp_1ReadVariableOpcond_input_2^cond*"
_output_shapes
:*
dtype02
Mul/ReadVariableOp_1x
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*"
_output_shapes
:2
Mul
AssignVariableOp_1AssignVariableOpcond_input_1Mul:z:0^Mul/ReadVariableOp^cond*
_output_shapes
 *
dtype02
AssignVariableOp_1y

group_depsNoOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_depsu
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2
group_deps_1p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	2
conv1d/ExpandDims®
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOpcond_input_1^AssignVariableOp_1*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1¸
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	*
squeeze_dims
2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2
ReluÄ
IdentityIdentityRelu:activations:0^AssignVariableOp^AssignVariableOp_1(^assert_greater_equal/Assert/AssertGuard^cond*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÝ	:::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2
condcond:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	
 
_user_specified_nameinputs:
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
: 
Þ

$__inference_signature_wrapper_317404
conv1d_input
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
identity¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallconv1d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

	*3
config_proto#!

GPU

CPU2*0,1,2,3J 8**
f%R#
!__inference__wrapped_model_3159452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿâ	::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	
&
_user_specified_nameconv1d_input:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ûf
È
X__inference_prune_low_magnitude_conv1d_2_layer_call_and_return_conditional_losses_316658

inputs
readvariableop_resource
cond_input_1
cond_input_2
cond_input_3#
biasadd_readvariableop_resource
identity¢AssignVariableOp¢AssignVariableOp_1¢'assert_greater_equal/Assert/AssertGuard¢condp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOpP
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
add/y\
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: 2
add
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOpA
updateNoOp^AssignVariableOp*
_output_shapes
 2
update­
#assert_greater_equal/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
: *
dtype0	2%
#assert_greater_equal/ReadVariableOpr
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
assert_greater_equal/yÅ
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2#
!assert_greater_equal/GreaterEqual{
assert_greater_equal/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
assert_greater_equal/Const
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/Const:output:0*
_output_shapes
: 2
assert_greater_equal/All
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2#
!assert_greater_equal/Assert/Const¶
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2%
#assert_greater_equal/Assert/Const_1·
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2%
#assert_greater_equal/Assert/Const_2ª
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2%
#assert_greater_equal/Assert/Const_3
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_316505*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_3165042)
'assert_greater_equal/Assert/AssertGuardÃ
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identityú
0polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	22
0polynomial_decay_pruning_schedule/ReadVariableOpÇ
'polynomial_decay_pruning_schedule/sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'polynomial_decay_pruning_schedule/sub/yâ
%polynomial_decay_pruning_schedule/subSub8polynomial_decay_pruning_schedule/ReadVariableOp:value:00polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/sub³
&polynomial_decay_pruning_schedule/CastCast)polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2(
&polynomial_decay_pruning_schedule/CastÒ
+polynomial_decay_pruning_schedule/truediv/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 * ¨G2-
+polynomial_decay_pruning_schedule/truediv/yä
)polynomial_decay_pruning_schedule/truedivRealDiv*polynomial_decay_pruning_schedule/Cast:y:04polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/truedivÒ
+polynomial_decay_pruning_schedule/Maximum/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *    2-
+polynomial_decay_pruning_schedule/Maximum/xç
)polynomial_decay_pruning_schedule/MaximumMaximum4polynomial_decay_pruning_schedule/Maximum/x:output:0-polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/MaximumÒ
+polynomial_decay_pruning_schedule/Minimum/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+polynomial_decay_pruning_schedule/Minimum/xç
)polynomial_decay_pruning_schedule/MinimumMinimum4polynomial_decay_pruning_schedule/Minimum/x:output:0-polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/MinimumÎ
)polynomial_decay_pruning_schedule/sub_1/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)polynomial_decay_pruning_schedule/sub_1/xÝ
'polynomial_decay_pruning_schedule/sub_1Sub2polynomial_decay_pruning_schedule/sub_1/x:output:0-polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2)
'polynomial_decay_pruning_schedule/sub_1Ê
'polynomial_decay_pruning_schedule/Pow/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  @@2)
'polynomial_decay_pruning_schedule/Pow/yÕ
%polynomial_decay_pruning_schedule/PowPow+polynomial_decay_pruning_schedule/sub_1:z:00polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/PowÊ
'polynomial_decay_pruning_schedule/Mul/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *¾2)
'polynomial_decay_pruning_schedule/Mul/xÓ
%polynomial_decay_pruning_schedule/MulMul0polynomial_decay_pruning_schedule/Mul/x:output:0)polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/MulÔ
,polynomial_decay_pruning_schedule/sparsity/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2.
,polynomial_decay_pruning_schedule/sparsity/yâ
*polynomial_decay_pruning_schedule/sparsityAdd)polynomial_decay_pruning_schedule/Mul:z:05polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2,
*polynomial_decay_pruning_schedule/sparsityÐ
GreaterEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOp
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqualÊ
LessEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqual
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
Less/x
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0	*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd¾
Sub/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOp
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
Sub

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorMod
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1û
condIfLogicalAnd_1:z:0readvariableop_resourcecond_input_1cond_input_2cond_input_3LogicalAnd_1:z:0^AssignVariableOp*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*$
else_branchR
cond_false_316562*
output_shapes
: *#
then_branchR
cond_true_3165612
condZ
cond/IdentityIdentitycond:output:0*
T0
*
_output_shapes
: 2
cond/Identityu
update_1NoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 2

update_1y
Mul/ReadVariableOpReadVariableOpcond_input_1*"
_output_shapes
:*
dtype02
Mul/ReadVariableOp
Mul/ReadVariableOp_1ReadVariableOpcond_input_2^cond*"
_output_shapes
:*
dtype02
Mul/ReadVariableOp_1x
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*"
_output_shapes
:2
Mul
AssignVariableOp_1AssignVariableOpcond_input_1Mul:z:0^Mul/ReadVariableOp^cond*
_output_shapes
 *
dtype02
AssignVariableOp_1y

group_depsNoOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_depsu
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2
group_deps_1p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	2
conv1d/ExpandDims®
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOpcond_input_1^AssignVariableOp_1*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1¸
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	*
squeeze_dims
2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2
ReluÄ
IdentityIdentityRelu:activations:0^AssignVariableOp^AssignVariableOp_1(^assert_greater_equal/Assert/AssertGuard^cond*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÝ	:::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2
condcond:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	
 
_user_specified_nameinputs:
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
: 
`
Å
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_319415

inputs
readvariableop_resource
cond_input_1
cond_input_2
cond_input_3#
biasadd_readvariableop_resource
identity¢AssignVariableOp¢AssignVariableOp_1¢'assert_greater_equal/Assert/AssertGuard¢condp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOpP
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
add/y\
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: 2
add
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOpA
updateNoOp^AssignVariableOp*
_output_shapes
 2
update­
#assert_greater_equal/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
: *
dtype0	2%
#assert_greater_equal/ReadVariableOpr
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
assert_greater_equal/yÅ
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2#
!assert_greater_equal/GreaterEqual{
assert_greater_equal/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
assert_greater_equal/Const
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/Const:output:0*
_output_shapes
: 2
assert_greater_equal/All
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2#
!assert_greater_equal/Assert/Const¶
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2%
#assert_greater_equal/Assert/Const_1·
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2%
#assert_greater_equal/Assert/Const_2ª
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2%
#assert_greater_equal/Assert/Const_3
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_319267*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_3192662)
'assert_greater_equal/Assert/AssertGuardÃ
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identityú
0polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	22
0polynomial_decay_pruning_schedule/ReadVariableOpÇ
'polynomial_decay_pruning_schedule/sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'polynomial_decay_pruning_schedule/sub/yâ
%polynomial_decay_pruning_schedule/subSub8polynomial_decay_pruning_schedule/ReadVariableOp:value:00polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/sub³
&polynomial_decay_pruning_schedule/CastCast)polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2(
&polynomial_decay_pruning_schedule/CastÒ
+polynomial_decay_pruning_schedule/truediv/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 * ¨G2-
+polynomial_decay_pruning_schedule/truediv/yä
)polynomial_decay_pruning_schedule/truedivRealDiv*polynomial_decay_pruning_schedule/Cast:y:04polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/truedivÒ
+polynomial_decay_pruning_schedule/Maximum/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *    2-
+polynomial_decay_pruning_schedule/Maximum/xç
)polynomial_decay_pruning_schedule/MaximumMaximum4polynomial_decay_pruning_schedule/Maximum/x:output:0-polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/MaximumÒ
+polynomial_decay_pruning_schedule/Minimum/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+polynomial_decay_pruning_schedule/Minimum/xç
)polynomial_decay_pruning_schedule/MinimumMinimum4polynomial_decay_pruning_schedule/Minimum/x:output:0-polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/MinimumÎ
)polynomial_decay_pruning_schedule/sub_1/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)polynomial_decay_pruning_schedule/sub_1/xÝ
'polynomial_decay_pruning_schedule/sub_1Sub2polynomial_decay_pruning_schedule/sub_1/x:output:0-polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2)
'polynomial_decay_pruning_schedule/sub_1Ê
'polynomial_decay_pruning_schedule/Pow/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  @@2)
'polynomial_decay_pruning_schedule/Pow/yÕ
%polynomial_decay_pruning_schedule/PowPow+polynomial_decay_pruning_schedule/sub_1:z:00polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/PowÊ
'polynomial_decay_pruning_schedule/Mul/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *¾2)
'polynomial_decay_pruning_schedule/Mul/xÓ
%polynomial_decay_pruning_schedule/MulMul0polynomial_decay_pruning_schedule/Mul/x:output:0)polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/MulÔ
,polynomial_decay_pruning_schedule/sparsity/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2.
,polynomial_decay_pruning_schedule/sparsity/yâ
*polynomial_decay_pruning_schedule/sparsityAdd)polynomial_decay_pruning_schedule/Mul:z:05polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2,
*polynomial_decay_pruning_schedule/sparsityÐ
GreaterEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOp
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqualÊ
LessEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqual
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
Less/x
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0	*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd¾
Sub/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOp
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
Sub

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorMod
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1û
condIfLogicalAnd_1:z:0readvariableop_resourcecond_input_1cond_input_2cond_input_3LogicalAnd_1:z:0^AssignVariableOp*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*$
else_branchR
cond_false_319324*
output_shapes
: *#
then_branchR
cond_true_3193232
condZ
cond/IdentityIdentitycond:output:0*
T0
*
_output_shapes
: 2
cond/Identityu
update_1NoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 2

update_1w
Mul/ReadVariableOpReadVariableOpcond_input_1* 
_output_shapes
:
*
dtype02
Mul/ReadVariableOp
Mul/ReadVariableOp_1ReadVariableOpcond_input_2^cond* 
_output_shapes
:
*
dtype02
Mul/ReadVariableOp_1v
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0* 
_output_shapes
:
2
Mul
AssignVariableOp_1AssignVariableOpcond_input_1Mul:z:0^Mul/ReadVariableOp^cond*
_output_shapes
 *
dtype02
AssignVariableOp_1y

group_depsNoOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_depsu
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2
group_deps_1
MatMul/ReadVariableOpReadVariableOpcond_input_1^AssignVariableOp_1* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoid¸
IdentityIdentitySigmoid:y:0^AssignVariableOp^AssignVariableOp_1(^assert_greater_equal/Assert/AssertGuard^cond*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ:::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2
condcond:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:
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
: 
©
Â
.prune_low_magnitude_conv1d_1_cond_false_317644
placeholder
placeholder_1
placeholder_2
placeholder_36
2identity_prune_low_magnitude_conv1d_1_logicaland_1


identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOp|
IdentityIdentity2identity_prune_low_magnitude_conv1d_1_logicaland_1^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
::::: : 

_output_shapes
: :
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
: 
ÿ
>
cond_false_316907
identity_logicaland_1


identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOp_
IdentityIdentityidentity_logicaland_1^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : 

_output_shapes
: 
©
Â
.prune_low_magnitude_conv1d_2_cond_false_317810
placeholder
placeholder_1
placeholder_2
placeholder_36
2identity_prune_low_magnitude_conv1d_2_logicaland_1


identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOp|
IdentityIdentity2identity_prune_low_magnitude_conv1d_2_logicaland_1^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
::::: : 

_output_shapes
: :
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
: 


+__inference_sequential_layer_call_fn_317365
conv1d_input
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
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallconv1d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

	*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3173382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿâ	::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	
&
_user_specified_nameconv1d_input:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

=
cond_true_316790
identity_logicaland_1


identity_1
6

group_depsNoOp*
_output_shapes
 2

group_depse
IdentityIdentityidentity_logicaland_1^group_deps*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : 

_output_shapes
: 
)

F__inference_sequential_layer_call_and_return_conditional_losses_317198
conv1d_input%
!prune_low_magnitude_conv1d_317167%
!prune_low_magnitude_conv1d_317169%
!prune_low_magnitude_conv1d_317171'
#prune_low_magnitude_conv1d_1_317174'
#prune_low_magnitude_conv1d_1_317176'
#prune_low_magnitude_conv1d_1_317178'
#prune_low_magnitude_conv1d_2_317181'
#prune_low_magnitude_conv1d_2_317183'
#prune_low_magnitude_conv1d_2_317185$
 prune_low_magnitude_dense_317190$
 prune_low_magnitude_dense_317192$
 prune_low_magnitude_dense_317194
identity¢2prune_low_magnitude_conv1d/StatefulPartitionedCall¢4prune_low_magnitude_conv1d_1/StatefulPartitionedCall¢4prune_low_magnitude_conv1d_2/StatefulPartitionedCall¢1prune_low_magnitude_dense/StatefulPartitionedCall
2prune_low_magnitude_conv1d/StatefulPartitionedCallStatefulPartitionedCallconv1d_input!prune_low_magnitude_conv1d_317167!prune_low_magnitude_conv1d_317169!prune_low_magnitude_conv1d_317171*
Tin
2*
Tout
2*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	*$
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*_
fZRX
V__inference_prune_low_magnitude_conv1d_layer_call_and_return_conditional_losses_31622024
2prune_low_magnitude_conv1d/StatefulPartitionedCall¿
4prune_low_magnitude_conv1d_1/StatefulPartitionedCallStatefulPartitionedCall;prune_low_magnitude_conv1d/StatefulPartitionedCall:output:0#prune_low_magnitude_conv1d_1_317174#prune_low_magnitude_conv1d_1_317176#prune_low_magnitude_conv1d_1_317178*
Tin
2*
Tout
2*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	*$
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*a
f\RZ
X__inference_prune_low_magnitude_conv1d_1_layer_call_and_return_conditional_losses_31644926
4prune_low_magnitude_conv1d_1/StatefulPartitionedCallÁ
4prune_low_magnitude_conv1d_2/StatefulPartitionedCallStatefulPartitionedCall=prune_low_magnitude_conv1d_1/StatefulPartitionedCall:output:0#prune_low_magnitude_conv1d_2_317181#prune_low_magnitude_conv1d_2_317183#prune_low_magnitude_conv1d_2_317185*
Tin
2*
Tout
2*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	*$
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*a
f\RZ
X__inference_prune_low_magnitude_conv1d_2_layer_call_and_return_conditional_losses_31667826
4prune_low_magnitude_conv1d_2/StatefulPartitionedCall°
+prune_low_magnitude_dropout/PartitionedCallPartitionedCall=prune_low_magnitude_conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	* 
_read_only_resource_inputs
 *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*`
f[RY
W__inference_prune_low_magnitude_dropout_layer_call_and_return_conditional_losses_3168162-
+prune_low_magnitude_dropout/PartitionedCall¤
+prune_low_magnitude_flatten/PartitionedCallPartitionedCall4prune_low_magnitude_dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*`
f[RY
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_3169272-
+prune_low_magnitude_flatten/PartitionedCall¡
1prune_low_magnitude_dense/StatefulPartitionedCallStatefulPartitionedCall4prune_low_magnitude_flatten/PartitionedCall:output:0 prune_low_magnitude_dense_317190 prune_low_magnitude_dense_317192 prune_low_magnitude_dense_317194*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*^
fYRW
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_31712423
1prune_low_magnitude_dense/StatefulPartitionedCallå
IdentityIdentity:prune_low_magnitude_dense/StatefulPartitionedCall:output:03^prune_low_magnitude_conv1d/StatefulPartitionedCall5^prune_low_magnitude_conv1d_1/StatefulPartitionedCall5^prune_low_magnitude_conv1d_2/StatefulPartitionedCall2^prune_low_magnitude_dense/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿâ	::::::::::::2h
2prune_low_magnitude_conv1d/StatefulPartitionedCall2prune_low_magnitude_conv1d/StatefulPartitionedCall2l
4prune_low_magnitude_conv1d_1/StatefulPartitionedCall4prune_low_magnitude_conv1d_1/StatefulPartitionedCall2l
4prune_low_magnitude_conv1d_2/StatefulPartitionedCall4prune_low_magnitude_conv1d_2/StatefulPartitionedCall2f
1prune_low_magnitude_dense/StatefulPartitionedCall1prune_low_magnitude_dense/StatefulPartitionedCall:Z V
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	
&
_user_specified_nameconv1d_input:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

u
W__inference_prune_low_magnitude_dropout_layer_call_and_return_conditional_losses_319132

inputs

identity_14
	no_updateNoOp*
_output_shapes
 2
	no_update8
no_update_1NoOp*
_output_shapes
 2
no_update_16

group_depsNoOp*
_output_shapes
 2

group_deps_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÙ	:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	
 
_user_specified_nameinputs


3assert_greater_equal_Assert_AssertGuard_true_316504%
!identity_assert_greater_equal_all

placeholder	
placeholder_1	

identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOpk
IdentityIdentity!identity_assert_greater_equal_all^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ù

<__inference_prune_low_magnitude_dropout_layer_call_fn_319139

inputs
unknown
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	* 
_read_only_resource_inputs
 *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*`
f[RY
W__inference_prune_low_magnitude_dropout_layer_call_and_return_conditional_losses_3168112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÙ	:22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	
 
_user_specified_nameinputs:

_output_shapes
: 


3assert_greater_equal_Assert_AssertGuard_true_316733%
!identity_assert_greater_equal_all

placeholder	
placeholder_1	

identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOpk
IdentityIdentity!identity_assert_greater_equal_all^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


3assert_greater_equal_Assert_AssertGuard_true_318401%
!identity_assert_greater_equal_all

placeholder	
placeholder_1	

identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOpk
IdentityIdentity!identity_assert_greater_equal_all^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


3assert_greater_equal_Assert_AssertGuard_true_316275%
!identity_assert_greater_equal_all

placeholder	
placeholder_1	

identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOpk
IdentityIdentity!identity_assert_greater_equal_all^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
K

cond_true_318890=
9polynomial_decay_pruning_schedule_readvariableop_resource+
'pruning_ops_abs_readvariableop_resource
assignvariableop_resource
assignvariableop_1_resource
identity_logicaland_1


identity_1
¢AssignVariableOp¢AssignVariableOp_1Ö
0polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	22
0polynomial_decay_pruning_schedule/ReadVariableOp
'polynomial_decay_pruning_schedule/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'polynomial_decay_pruning_schedule/sub/yâ
%polynomial_decay_pruning_schedule/subSub8polynomial_decay_pruning_schedule/ReadVariableOp:value:00polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/sub³
&polynomial_decay_pruning_schedule/CastCast)polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2(
&polynomial_decay_pruning_schedule/Cast
+polynomial_decay_pruning_schedule/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ¨G2-
+polynomial_decay_pruning_schedule/truediv/yä
)polynomial_decay_pruning_schedule/truedivRealDiv*polynomial_decay_pruning_schedule/Cast:y:04polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/truediv
+polynomial_decay_pruning_schedule/Maximum/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+polynomial_decay_pruning_schedule/Maximum/xç
)polynomial_decay_pruning_schedule/MaximumMaximum4polynomial_decay_pruning_schedule/Maximum/x:output:0-polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/Maximum
+polynomial_decay_pruning_schedule/Minimum/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+polynomial_decay_pruning_schedule/Minimum/xç
)polynomial_decay_pruning_schedule/MinimumMinimum4polynomial_decay_pruning_schedule/Minimum/x:output:0-polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/Minimum
)polynomial_decay_pruning_schedule/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)polynomial_decay_pruning_schedule/sub_1/xÝ
'polynomial_decay_pruning_schedule/sub_1Sub2polynomial_decay_pruning_schedule/sub_1/x:output:0-polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2)
'polynomial_decay_pruning_schedule/sub_1
'polynomial_decay_pruning_schedule/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2)
'polynomial_decay_pruning_schedule/Pow/yÕ
%polynomial_decay_pruning_schedule/PowPow+polynomial_decay_pruning_schedule/sub_1:z:00polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/Pow
'polynomial_decay_pruning_schedule/Mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¾2)
'polynomial_decay_pruning_schedule/Mul/xÓ
%polynomial_decay_pruning_schedule/MulMul0polynomial_decay_pruning_schedule/Mul/x:output:0)polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/Mul¡
,polynomial_decay_pruning_schedule/sparsity/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2.
,polynomial_decay_pruning_schedule/sparsity/yâ
*polynomial_decay_pruning_schedule/sparsityAdd)polynomial_decay_pruning_schedule/Mul:z:05polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2,
*polynomial_decay_pruning_schedule/sparsity¬
GreaterEqual/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOpb
GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqual¦
LessEqual/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp^
LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqualT
Less/xConst*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
Less/xR
Less/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0	*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd
Sub/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOpP
Sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
SubZ

FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorModT
Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1¬
pruning_ops/Abs/ReadVariableOpReadVariableOp'pruning_ops_abs_readvariableop_resource*"
_output_shapes
:*
dtype02 
pruning_ops/Abs/ReadVariableOp~
pruning_ops/AbsAbs&pruning_ops/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:2
pruning_ops/Absg
pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value
B :
2
pruning_ops/Sizew
pruning_ops/CastCastpruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
pruning_ops/Castk
pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
pruning_ops/sub/x
pruning_ops/subSubpruning_ops/sub/x:output:0.polynomial_decay_pruning_schedule/sparsity:z:0*
T0*
_output_shapes
: 2
pruning_ops/subu
pruning_ops/mulMulpruning_ops/Cast:y:0pruning_ops/sub:z:0*
T0*
_output_shapes
: 2
pruning_ops/mule
pruning_ops/RoundRoundpruning_ops/mul:z:0*
T0*
_output_shapes
: 2
pruning_ops/Rounds
pruning_ops/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
pruning_ops/Maximum/y
pruning_ops/MaximumMaximumpruning_ops/Round:y:0pruning_ops/Maximum/y:output:0*
T0*
_output_shapes
: 2
pruning_ops/Maximumy
pruning_ops/Cast_1Castpruning_ops/Maximum:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
pruning_ops/Cast_1
pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
pruning_ops/Reshape/shape
pruning_ops/ReshapeReshapepruning_ops/Abs:y:0"pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes	
:
2
pruning_ops/Reshapek
pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value
B :
2
pruning_ops/Size_1
pruning_ops/TopKV2TopKV2pruning_ops/Reshape:output:0pruning_ops/Size_1:output:0*
T0*"
_output_shapes
:
:
2
pruning_ops/TopKV2l
pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
pruning_ops/sub_1/y
pruning_ops/sub_1Subpruning_ops/Cast_1:y:0pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: 2
pruning_ops/sub_1x
pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
pruning_ops/GatherV2/axisÔ
pruning_ops/GatherV2GatherV2pruning_ops/TopKV2:values:0pruning_ops/sub_1:z:0"pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: 2
pruning_ops/GatherV2¥
pruning_ops/GreaterEqualGreaterEqualpruning_ops/Abs:y:0pruning_ops/GatherV2:output:0*
T0*"
_output_shapes
:2
pruning_ops/GreaterEqual
pruning_ops/Cast_2Castpruning_ops/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
:2
pruning_ops/Cast_2
AssignVariableOpAssignVariableOpassignvariableop_resourcepruning_ops/Cast_2:y:0*
_output_shapes
 *
dtype02
AssignVariableOp
AssignVariableOp_1AssignVariableOpassignvariableop_1_resourcepruning_ops/GatherV2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1

group_depsNoOp^AssignVariableOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_depse
IdentityIdentityidentity_logicaland_1^group_deps*
T0
*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^AssignVariableOp^AssignVariableOp_1*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
::::: 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_1: 

_output_shapes
: :
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
: 
ó
ó
X__inference_prune_low_magnitude_conv1d_2_layer_call_and_return_conditional_losses_316678

inputs
mul_readvariableop_resource!
mul_readvariableop_1_resource#
biasadd_readvariableop_resource
identity¢AssignVariableOp4
	no_updateNoOp*
_output_shapes
 2
	no_update8
no_update_1NoOp*
_output_shapes
 2
no_update_1
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*"
_output_shapes
:*
dtype02
Mul/ReadVariableOp
Mul/ReadVariableOp_1ReadVariableOpmul_readvariableop_1_resource*"
_output_shapes
:*
dtype02
Mul/ReadVariableOp_1x
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*"
_output_shapes
:2
Mul
AssignVariableOpAssignVariableOpmul_readvariableop_resourceMul:z:0^Mul/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpw

group_depsNoOp^AssignVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_depsu
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2
group_deps_1p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	2
conv1d/ExpandDims»
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1¸
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	*
squeeze_dims
2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2
Relu~
IdentityIdentityRelu:activations:0^AssignVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÝ	:::2$
AssignVariableOpAssignVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ùf
Æ
V__inference_prune_low_magnitude_conv1d_layer_call_and_return_conditional_losses_318555

inputs
readvariableop_resource
cond_input_1
cond_input_2
cond_input_3#
biasadd_readvariableop_resource
identity¢AssignVariableOp¢AssignVariableOp_1¢'assert_greater_equal/Assert/AssertGuard¢condp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOpP
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
add/y\
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: 2
add
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOpA
updateNoOp^AssignVariableOp*
_output_shapes
 2
update­
#assert_greater_equal/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
: *
dtype0	2%
#assert_greater_equal/ReadVariableOpr
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
assert_greater_equal/yÅ
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2#
!assert_greater_equal/GreaterEqual{
assert_greater_equal/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
assert_greater_equal/Const
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/Const:output:0*
_output_shapes
: 2
assert_greater_equal/All
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2#
!assert_greater_equal/Assert/Const¶
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2%
#assert_greater_equal/Assert/Const_1·
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2%
#assert_greater_equal/Assert/Const_2ª
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2%
#assert_greater_equal/Assert/Const_3
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_318402*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_3184012)
'assert_greater_equal/Assert/AssertGuardÃ
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identityú
0polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	22
0polynomial_decay_pruning_schedule/ReadVariableOpÇ
'polynomial_decay_pruning_schedule/sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'polynomial_decay_pruning_schedule/sub/yâ
%polynomial_decay_pruning_schedule/subSub8polynomial_decay_pruning_schedule/ReadVariableOp:value:00polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/sub³
&polynomial_decay_pruning_schedule/CastCast)polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2(
&polynomial_decay_pruning_schedule/CastÒ
+polynomial_decay_pruning_schedule/truediv/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 * ¨G2-
+polynomial_decay_pruning_schedule/truediv/yä
)polynomial_decay_pruning_schedule/truedivRealDiv*polynomial_decay_pruning_schedule/Cast:y:04polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/truedivÒ
+polynomial_decay_pruning_schedule/Maximum/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *    2-
+polynomial_decay_pruning_schedule/Maximum/xç
)polynomial_decay_pruning_schedule/MaximumMaximum4polynomial_decay_pruning_schedule/Maximum/x:output:0-polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/MaximumÒ
+polynomial_decay_pruning_schedule/Minimum/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+polynomial_decay_pruning_schedule/Minimum/xç
)polynomial_decay_pruning_schedule/MinimumMinimum4polynomial_decay_pruning_schedule/Minimum/x:output:0-polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/MinimumÎ
)polynomial_decay_pruning_schedule/sub_1/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)polynomial_decay_pruning_schedule/sub_1/xÝ
'polynomial_decay_pruning_schedule/sub_1Sub2polynomial_decay_pruning_schedule/sub_1/x:output:0-polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2)
'polynomial_decay_pruning_schedule/sub_1Ê
'polynomial_decay_pruning_schedule/Pow/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  @@2)
'polynomial_decay_pruning_schedule/Pow/yÕ
%polynomial_decay_pruning_schedule/PowPow+polynomial_decay_pruning_schedule/sub_1:z:00polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/PowÊ
'polynomial_decay_pruning_schedule/Mul/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *¾2)
'polynomial_decay_pruning_schedule/Mul/xÓ
%polynomial_decay_pruning_schedule/MulMul0polynomial_decay_pruning_schedule/Mul/x:output:0)polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/MulÔ
,polynomial_decay_pruning_schedule/sparsity/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2.
,polynomial_decay_pruning_schedule/sparsity/yâ
*polynomial_decay_pruning_schedule/sparsityAdd)polynomial_decay_pruning_schedule/Mul:z:05polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2,
*polynomial_decay_pruning_schedule/sparsityÐ
GreaterEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOp
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqualÊ
LessEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqual
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
Less/x
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0	*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd¾
Sub/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOp
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
Sub

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorMod
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1û
condIfLogicalAnd_1:z:0readvariableop_resourcecond_input_1cond_input_2cond_input_3LogicalAnd_1:z:0^AssignVariableOp*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*$
else_branchR
cond_false_318459*
output_shapes
: *#
then_branchR
cond_true_3184582
condZ
cond/IdentityIdentitycond:output:0*
T0
*
_output_shapes
: 2
cond/Identityu
update_1NoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 2

update_1y
Mul/ReadVariableOpReadVariableOpcond_input_1*"
_output_shapes
:*
dtype02
Mul/ReadVariableOp
Mul/ReadVariableOp_1ReadVariableOpcond_input_2^cond*"
_output_shapes
:*
dtype02
Mul/ReadVariableOp_1x
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*"
_output_shapes
:2
Mul
AssignVariableOp_1AssignVariableOpcond_input_1Mul:z:0^Mul/ReadVariableOp^cond*
_output_shapes
 *
dtype02
AssignVariableOp_1y

group_depsNoOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_depsu
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2
group_deps_1p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	2
conv1d/ExpandDims®
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOpcond_input_1^AssignVariableOp_1*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1¸
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	*
squeeze_dims
2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	2
ReluÄ
IdentityIdentityRelu:activations:0^AssignVariableOp^AssignVariableOp_1(^assert_greater_equal/Assert/AssertGuard^cond*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿâ	:::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2
condcond:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	
 
_user_specified_nameinputs:
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
: 
ã
Ê
4assert_greater_equal_Assert_AssertGuard_false_319050#
assert_assert_greater_equal_all
.
*assert_assert_greater_equal_readvariableop	!
assert_assert_greater_equal_y	

identity_1
¢Assertî
Assert/data_0Const*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2
Assert/data_0
Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2
Assert/data_1
Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2
Assert/data_2~
Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2
Assert/data_4
AssertAssertassert_assert_greater_equal_allAssert/data_0:output:0Assert/data_1:output:0Assert/data_2:output:0*assert_assert_greater_equal_readvariableopAssert/data_4:output:0assert_assert_greater_equal_y*
T

2		*
_output_shapes
 2
Assertk
IdentityIdentityassert_assert_greater_equal_all^Assert*
T0
*
_output_shapes
: 2

Identitya

Identity_1IdentityIdentity:output:0^Assert*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : 2
AssertAssert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
õ

;__inference_prune_low_magnitude_conv1d_layer_call_fn_318601

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	*$
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*_
fZRX
V__inference_prune_low_magnitude_conv1d_layer_call_and_return_conditional_losses_3162202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿâ	:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
³
~
)__inference_conv1d_2_layer_call_fn_316026

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_3160162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ý
º
Pprune_low_magnitude_dropout_assert_greater_equal_Assert_AssertGuard_false_317919?
;assert_prune_low_magnitude_dropout_assert_greater_equal_all
J
Fassert_prune_low_magnitude_dropout_assert_greater_equal_readvariableop	=
9assert_prune_low_magnitude_dropout_assert_greater_equal_y	

identity_1
¢Assertî
Assert/data_0Const*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2
Assert/data_0
Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2
Assert/data_1§
Assert/data_2Const*
_output_shapes
: *
dtype0*Y
valuePBN BHx (prune_low_magnitude_dropout/assert_greater_equal/ReadVariableOp:0) = 2
Assert/data_2
Assert/data_4Const*
_output_shapes
: *
dtype0*L
valueCBA B;y (prune_low_magnitude_dropout/assert_greater_equal/y:0) = 2
Assert/data_4á
AssertAssert;assert_prune_low_magnitude_dropout_assert_greater_equal_allAssert/data_0:output:0Assert/data_1:output:0Assert/data_2:output:0Fassert_prune_low_magnitude_dropout_assert_greater_equal_readvariableopAssert/data_4:output:09assert_prune_low_magnitude_dropout_assert_greater_equal_y*
T

2		*
_output_shapes
 2
Assert
IdentityIdentity;assert_prune_low_magnitude_dropout_assert_greater_equal_all^Assert*
T0
*
_output_shapes
: 2

Identitya

Identity_1IdentityIdentity:output:0^Assert*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : 2
AssertAssert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
³
~
)__inference_conv1d_1_layer_call_fn_315999

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_3159892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
°5


F__inference_sequential_layer_call_and_return_conditional_losses_317255

inputs%
!prune_low_magnitude_conv1d_317204%
!prune_low_magnitude_conv1d_317206%
!prune_low_magnitude_conv1d_317208%
!prune_low_magnitude_conv1d_317210%
!prune_low_magnitude_conv1d_317212'
#prune_low_magnitude_conv1d_1_317215'
#prune_low_magnitude_conv1d_1_317217'
#prune_low_magnitude_conv1d_1_317219'
#prune_low_magnitude_conv1d_1_317221'
#prune_low_magnitude_conv1d_1_317223'
#prune_low_magnitude_conv1d_2_317226'
#prune_low_magnitude_conv1d_2_317228'
#prune_low_magnitude_conv1d_2_317230'
#prune_low_magnitude_conv1d_2_317232'
#prune_low_magnitude_conv1d_2_317234&
"prune_low_magnitude_dropout_317237&
"prune_low_magnitude_flatten_317240$
 prune_low_magnitude_dense_317243$
 prune_low_magnitude_dense_317245$
 prune_low_magnitude_dense_317247$
 prune_low_magnitude_dense_317249$
 prune_low_magnitude_dense_317251
identity¢2prune_low_magnitude_conv1d/StatefulPartitionedCall¢4prune_low_magnitude_conv1d_1/StatefulPartitionedCall¢4prune_low_magnitude_conv1d_2/StatefulPartitionedCall¢1prune_low_magnitude_dense/StatefulPartitionedCall¢3prune_low_magnitude_dropout/StatefulPartitionedCall¢3prune_low_magnitude_flatten/StatefulPartitionedCallÅ
2prune_low_magnitude_conv1d/StatefulPartitionedCallStatefulPartitionedCallinputs!prune_low_magnitude_conv1d_317204!prune_low_magnitude_conv1d_317206!prune_low_magnitude_conv1d_317208!prune_low_magnitude_conv1d_317210!prune_low_magnitude_conv1d_317212*
Tin

2*
Tout
2*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	*#
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*_
fZRX
V__inference_prune_low_magnitude_conv1d_layer_call_and_return_conditional_losses_31620024
2prune_low_magnitude_conv1d/StatefulPartitionedCall
4prune_low_magnitude_conv1d_1/StatefulPartitionedCallStatefulPartitionedCall;prune_low_magnitude_conv1d/StatefulPartitionedCall:output:0#prune_low_magnitude_conv1d_1_317215#prune_low_magnitude_conv1d_1_317217#prune_low_magnitude_conv1d_1_317219#prune_low_magnitude_conv1d_1_317221#prune_low_magnitude_conv1d_1_317223*
Tin

2*
Tout
2*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	*#
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*a
f\RZ
X__inference_prune_low_magnitude_conv1d_1_layer_call_and_return_conditional_losses_31642926
4prune_low_magnitude_conv1d_1/StatefulPartitionedCall
4prune_low_magnitude_conv1d_2/StatefulPartitionedCallStatefulPartitionedCall=prune_low_magnitude_conv1d_1/StatefulPartitionedCall:output:0#prune_low_magnitude_conv1d_2_317226#prune_low_magnitude_conv1d_2_317228#prune_low_magnitude_conv1d_2_317230#prune_low_magnitude_conv1d_2_317232#prune_low_magnitude_conv1d_2_317234*
Tin

2*
Tout
2*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	*#
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*a
f\RZ
X__inference_prune_low_magnitude_conv1d_2_layer_call_and_return_conditional_losses_31665826
4prune_low_magnitude_conv1d_2/StatefulPartitionedCallí
3prune_low_magnitude_dropout/StatefulPartitionedCallStatefulPartitionedCall=prune_low_magnitude_conv1d_2/StatefulPartitionedCall:output:0"prune_low_magnitude_dropout_317237*
Tin
2*
Tout
2*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	* 
_read_only_resource_inputs
 *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*`
f[RY
W__inference_prune_low_magnitude_dropout_layer_call_and_return_conditional_losses_31681125
3prune_low_magnitude_dropout/StatefulPartitionedCallé
3prune_low_magnitude_flatten/StatefulPartitionedCallStatefulPartitionedCall<prune_low_magnitude_dropout/StatefulPartitionedCall:output:0"prune_low_magnitude_flatten_317240*
Tin
2*
Tout
2*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*`
f[RY
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_31692125
3prune_low_magnitude_flatten/StatefulPartitionedCallî
1prune_low_magnitude_dense/StatefulPartitionedCallStatefulPartitionedCall<prune_low_magnitude_flatten/StatefulPartitionedCall:output:0 prune_low_magnitude_dense_317243 prune_low_magnitude_dense_317245 prune_low_magnitude_dense_317247 prune_low_magnitude_dense_317249 prune_low_magnitude_dense_317251*
Tin

2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*^
fYRW
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_31710923
1prune_low_magnitude_dense/StatefulPartitionedCallÑ
IdentityIdentity:prune_low_magnitude_dense/StatefulPartitionedCall:output:03^prune_low_magnitude_conv1d/StatefulPartitionedCall5^prune_low_magnitude_conv1d_1/StatefulPartitionedCall5^prune_low_magnitude_conv1d_2/StatefulPartitionedCall2^prune_low_magnitude_dense/StatefulPartitionedCall4^prune_low_magnitude_dropout/StatefulPartitionedCall4^prune_low_magnitude_flatten/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesr
p:ÿÿÿÿÿÿÿÿÿâ	::::::::::::::::::::::2h
2prune_low_magnitude_conv1d/StatefulPartitionedCall2prune_low_magnitude_conv1d/StatefulPartitionedCall2l
4prune_low_magnitude_conv1d_1/StatefulPartitionedCall4prune_low_magnitude_conv1d_1/StatefulPartitionedCall2l
4prune_low_magnitude_conv1d_2/StatefulPartitionedCall4prune_low_magnitude_conv1d_2/StatefulPartitionedCall2f
1prune_low_magnitude_dense/StatefulPartitionedCall1prune_low_magnitude_dense/StatefulPartitionedCall2j
3prune_low_magnitude_dropout/StatefulPartitionedCall3prune_low_magnitude_dropout/StatefulPartitionedCall2j
3prune_low_magnitude_flatten/StatefulPartitionedCall3prune_low_magnitude_flatten/StatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	
 
_user_specified_nameinputs:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¦
¢	
!__inference__wrapped_model_315945
conv1d_inputE
Asequential_prune_low_magnitude_conv1d_mul_readvariableop_resourceG
Csequential_prune_low_magnitude_conv1d_mul_readvariableop_1_resourceI
Esequential_prune_low_magnitude_conv1d_biasadd_readvariableop_resourceG
Csequential_prune_low_magnitude_conv1d_1_mul_readvariableop_resourceI
Esequential_prune_low_magnitude_conv1d_1_mul_readvariableop_1_resourceK
Gsequential_prune_low_magnitude_conv1d_1_biasadd_readvariableop_resourceG
Csequential_prune_low_magnitude_conv1d_2_mul_readvariableop_resourceI
Esequential_prune_low_magnitude_conv1d_2_mul_readvariableop_1_resourceK
Gsequential_prune_low_magnitude_conv1d_2_biasadd_readvariableop_resourceD
@sequential_prune_low_magnitude_dense_mul_readvariableop_resourceF
Bsequential_prune_low_magnitude_dense_mul_readvariableop_1_resourceH
Dsequential_prune_low_magnitude_dense_biasadd_readvariableop_resource
identity¢6sequential/prune_low_magnitude_conv1d/AssignVariableOp¢8sequential/prune_low_magnitude_conv1d_1/AssignVariableOp¢8sequential/prune_low_magnitude_conv1d_2/AssignVariableOp¢5sequential/prune_low_magnitude_dense/AssignVariableOp
/sequential/prune_low_magnitude_conv1d/no_updateNoOp*
_output_shapes
 21
/sequential/prune_low_magnitude_conv1d/no_update
1sequential/prune_low_magnitude_conv1d/no_update_1NoOp*
_output_shapes
 23
1sequential/prune_low_magnitude_conv1d/no_update_1ú
8sequential/prune_low_magnitude_conv1d/Mul/ReadVariableOpReadVariableOpAsequential_prune_low_magnitude_conv1d_mul_readvariableop_resource*"
_output_shapes
:*
dtype02:
8sequential/prune_low_magnitude_conv1d/Mul/ReadVariableOp
:sequential/prune_low_magnitude_conv1d/Mul/ReadVariableOp_1ReadVariableOpCsequential_prune_low_magnitude_conv1d_mul_readvariableop_1_resource*"
_output_shapes
:*
dtype02<
:sequential/prune_low_magnitude_conv1d/Mul/ReadVariableOp_1
)sequential/prune_low_magnitude_conv1d/MulMul@sequential/prune_low_magnitude_conv1d/Mul/ReadVariableOp:value:0Bsequential/prune_low_magnitude_conv1d/Mul/ReadVariableOp_1:value:0*
T0*"
_output_shapes
:2+
)sequential/prune_low_magnitude_conv1d/MulÔ
6sequential/prune_low_magnitude_conv1d/AssignVariableOpAssignVariableOpAsequential_prune_low_magnitude_conv1d_mul_readvariableop_resource-sequential/prune_low_magnitude_conv1d/Mul:z:09^sequential/prune_low_magnitude_conv1d/Mul/ReadVariableOp*
_output_shapes
 *
dtype028
6sequential/prune_low_magnitude_conv1d/AssignVariableOpé
0sequential/prune_low_magnitude_conv1d/group_depsNoOp7^sequential/prune_low_magnitude_conv1d/AssignVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 22
0sequential/prune_low_magnitude_conv1d/group_depsç
2sequential/prune_low_magnitude_conv1d/group_deps_1NoOp1^sequential/prune_low_magnitude_conv1d/group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 24
2sequential/prune_low_magnitude_conv1d/group_deps_1¼
;sequential/prune_low_magnitude_conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2=
;sequential/prune_low_magnitude_conv1d/conv1d/ExpandDims/dim
7sequential/prune_low_magnitude_conv1d/conv1d/ExpandDims
ExpandDimsconv1d_inputDsequential/prune_low_magnitude_conv1d/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	29
7sequential/prune_low_magnitude_conv1d/conv1d/ExpandDimsÓ
Hsequential/prune_low_magnitude_conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAsequential_prune_low_magnitude_conv1d_mul_readvariableop_resource7^sequential/prune_low_magnitude_conv1d/AssignVariableOp*"
_output_shapes
:*
dtype02J
Hsequential/prune_low_magnitude_conv1d/conv1d/ExpandDims_1/ReadVariableOpÀ
=sequential/prune_low_magnitude_conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2?
=sequential/prune_low_magnitude_conv1d/conv1d/ExpandDims_1/dimÏ
9sequential/prune_low_magnitude_conv1d/conv1d/ExpandDims_1
ExpandDimsPsequential/prune_low_magnitude_conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0Fsequential/prune_low_magnitude_conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2;
9sequential/prune_low_magnitude_conv1d/conv1d/ExpandDims_1Ð
,sequential/prune_low_magnitude_conv1d/conv1dConv2D@sequential/prune_low_magnitude_conv1d/conv1d/ExpandDims:output:0Bsequential/prune_low_magnitude_conv1d/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	*
paddingVALID*
strides
2.
,sequential/prune_low_magnitude_conv1d/conv1dü
4sequential/prune_low_magnitude_conv1d/conv1d/SqueezeSqueeze5sequential/prune_low_magnitude_conv1d/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	*
squeeze_dims
26
4sequential/prune_low_magnitude_conv1d/conv1d/Squeezeþ
<sequential/prune_low_magnitude_conv1d/BiasAdd/ReadVariableOpReadVariableOpEsequential_prune_low_magnitude_conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02>
<sequential/prune_low_magnitude_conv1d/BiasAdd/ReadVariableOp¥
-sequential/prune_low_magnitude_conv1d/BiasAddBiasAdd=sequential/prune_low_magnitude_conv1d/conv1d/Squeeze:output:0Dsequential/prune_low_magnitude_conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	2/
-sequential/prune_low_magnitude_conv1d/BiasAddÏ
*sequential/prune_low_magnitude_conv1d/ReluRelu6sequential/prune_low_magnitude_conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	2,
*sequential/prune_low_magnitude_conv1d/Relu
1sequential/prune_low_magnitude_conv1d_1/no_updateNoOp*
_output_shapes
 23
1sequential/prune_low_magnitude_conv1d_1/no_update
3sequential/prune_low_magnitude_conv1d_1/no_update_1NoOp*
_output_shapes
 25
3sequential/prune_low_magnitude_conv1d_1/no_update_1
:sequential/prune_low_magnitude_conv1d_1/Mul/ReadVariableOpReadVariableOpCsequential_prune_low_magnitude_conv1d_1_mul_readvariableop_resource*"
_output_shapes
:*
dtype02<
:sequential/prune_low_magnitude_conv1d_1/Mul/ReadVariableOp
<sequential/prune_low_magnitude_conv1d_1/Mul/ReadVariableOp_1ReadVariableOpEsequential_prune_low_magnitude_conv1d_1_mul_readvariableop_1_resource*"
_output_shapes
:*
dtype02>
<sequential/prune_low_magnitude_conv1d_1/Mul/ReadVariableOp_1
+sequential/prune_low_magnitude_conv1d_1/MulMulBsequential/prune_low_magnitude_conv1d_1/Mul/ReadVariableOp:value:0Dsequential/prune_low_magnitude_conv1d_1/Mul/ReadVariableOp_1:value:0*
T0*"
_output_shapes
:2-
+sequential/prune_low_magnitude_conv1d_1/MulÞ
8sequential/prune_low_magnitude_conv1d_1/AssignVariableOpAssignVariableOpCsequential_prune_low_magnitude_conv1d_1_mul_readvariableop_resource/sequential/prune_low_magnitude_conv1d_1/Mul:z:0;^sequential/prune_low_magnitude_conv1d_1/Mul/ReadVariableOp*
_output_shapes
 *
dtype02:
8sequential/prune_low_magnitude_conv1d_1/AssignVariableOpï
2sequential/prune_low_magnitude_conv1d_1/group_depsNoOp9^sequential/prune_low_magnitude_conv1d_1/AssignVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 24
2sequential/prune_low_magnitude_conv1d_1/group_depsí
4sequential/prune_low_magnitude_conv1d_1/group_deps_1NoOp3^sequential/prune_low_magnitude_conv1d_1/group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 26
4sequential/prune_low_magnitude_conv1d_1/group_deps_1À
=sequential/prune_low_magnitude_conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2?
=sequential/prune_low_magnitude_conv1d_1/conv1d/ExpandDims/dimÁ
9sequential/prune_low_magnitude_conv1d_1/conv1d/ExpandDims
ExpandDims8sequential/prune_low_magnitude_conv1d/Relu:activations:0Fsequential/prune_low_magnitude_conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	2;
9sequential/prune_low_magnitude_conv1d_1/conv1d/ExpandDimsÛ
Jsequential/prune_low_magnitude_conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCsequential_prune_low_magnitude_conv1d_1_mul_readvariableop_resource9^sequential/prune_low_magnitude_conv1d_1/AssignVariableOp*"
_output_shapes
:*
dtype02L
Jsequential/prune_low_magnitude_conv1d_1/conv1d/ExpandDims_1/ReadVariableOpÄ
?sequential/prune_low_magnitude_conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2A
?sequential/prune_low_magnitude_conv1d_1/conv1d/ExpandDims_1/dim×
;sequential/prune_low_magnitude_conv1d_1/conv1d/ExpandDims_1
ExpandDimsRsequential/prune_low_magnitude_conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0Hsequential/prune_low_magnitude_conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2=
;sequential/prune_low_magnitude_conv1d_1/conv1d/ExpandDims_1Ø
.sequential/prune_low_magnitude_conv1d_1/conv1dConv2DBsequential/prune_low_magnitude_conv1d_1/conv1d/ExpandDims:output:0Dsequential/prune_low_magnitude_conv1d_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	*
paddingVALID*
strides
20
.sequential/prune_low_magnitude_conv1d_1/conv1d
6sequential/prune_low_magnitude_conv1d_1/conv1d/SqueezeSqueeze7sequential/prune_low_magnitude_conv1d_1/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	*
squeeze_dims
28
6sequential/prune_low_magnitude_conv1d_1/conv1d/Squeeze
>sequential/prune_low_magnitude_conv1d_1/BiasAdd/ReadVariableOpReadVariableOpGsequential_prune_low_magnitude_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>sequential/prune_low_magnitude_conv1d_1/BiasAdd/ReadVariableOp­
/sequential/prune_low_magnitude_conv1d_1/BiasAddBiasAdd?sequential/prune_low_magnitude_conv1d_1/conv1d/Squeeze:output:0Fsequential/prune_low_magnitude_conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	21
/sequential/prune_low_magnitude_conv1d_1/BiasAddÕ
,sequential/prune_low_magnitude_conv1d_1/ReluRelu8sequential/prune_low_magnitude_conv1d_1/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	2.
,sequential/prune_low_magnitude_conv1d_1/Relu
1sequential/prune_low_magnitude_conv1d_2/no_updateNoOp*
_output_shapes
 23
1sequential/prune_low_magnitude_conv1d_2/no_update
3sequential/prune_low_magnitude_conv1d_2/no_update_1NoOp*
_output_shapes
 25
3sequential/prune_low_magnitude_conv1d_2/no_update_1
:sequential/prune_low_magnitude_conv1d_2/Mul/ReadVariableOpReadVariableOpCsequential_prune_low_magnitude_conv1d_2_mul_readvariableop_resource*"
_output_shapes
:*
dtype02<
:sequential/prune_low_magnitude_conv1d_2/Mul/ReadVariableOp
<sequential/prune_low_magnitude_conv1d_2/Mul/ReadVariableOp_1ReadVariableOpEsequential_prune_low_magnitude_conv1d_2_mul_readvariableop_1_resource*"
_output_shapes
:*
dtype02>
<sequential/prune_low_magnitude_conv1d_2/Mul/ReadVariableOp_1
+sequential/prune_low_magnitude_conv1d_2/MulMulBsequential/prune_low_magnitude_conv1d_2/Mul/ReadVariableOp:value:0Dsequential/prune_low_magnitude_conv1d_2/Mul/ReadVariableOp_1:value:0*
T0*"
_output_shapes
:2-
+sequential/prune_low_magnitude_conv1d_2/MulÞ
8sequential/prune_low_magnitude_conv1d_2/AssignVariableOpAssignVariableOpCsequential_prune_low_magnitude_conv1d_2_mul_readvariableop_resource/sequential/prune_low_magnitude_conv1d_2/Mul:z:0;^sequential/prune_low_magnitude_conv1d_2/Mul/ReadVariableOp*
_output_shapes
 *
dtype02:
8sequential/prune_low_magnitude_conv1d_2/AssignVariableOpï
2sequential/prune_low_magnitude_conv1d_2/group_depsNoOp9^sequential/prune_low_magnitude_conv1d_2/AssignVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 24
2sequential/prune_low_magnitude_conv1d_2/group_depsí
4sequential/prune_low_magnitude_conv1d_2/group_deps_1NoOp3^sequential/prune_low_magnitude_conv1d_2/group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 26
4sequential/prune_low_magnitude_conv1d_2/group_deps_1À
=sequential/prune_low_magnitude_conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2?
=sequential/prune_low_magnitude_conv1d_2/conv1d/ExpandDims/dimÃ
9sequential/prune_low_magnitude_conv1d_2/conv1d/ExpandDims
ExpandDims:sequential/prune_low_magnitude_conv1d_1/Relu:activations:0Fsequential/prune_low_magnitude_conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	2;
9sequential/prune_low_magnitude_conv1d_2/conv1d/ExpandDimsÛ
Jsequential/prune_low_magnitude_conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCsequential_prune_low_magnitude_conv1d_2_mul_readvariableop_resource9^sequential/prune_low_magnitude_conv1d_2/AssignVariableOp*"
_output_shapes
:*
dtype02L
Jsequential/prune_low_magnitude_conv1d_2/conv1d/ExpandDims_1/ReadVariableOpÄ
?sequential/prune_low_magnitude_conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2A
?sequential/prune_low_magnitude_conv1d_2/conv1d/ExpandDims_1/dim×
;sequential/prune_low_magnitude_conv1d_2/conv1d/ExpandDims_1
ExpandDimsRsequential/prune_low_magnitude_conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0Hsequential/prune_low_magnitude_conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2=
;sequential/prune_low_magnitude_conv1d_2/conv1d/ExpandDims_1Ø
.sequential/prune_low_magnitude_conv1d_2/conv1dConv2DBsequential/prune_low_magnitude_conv1d_2/conv1d/ExpandDims:output:0Dsequential/prune_low_magnitude_conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	*
paddingVALID*
strides
20
.sequential/prune_low_magnitude_conv1d_2/conv1d
6sequential/prune_low_magnitude_conv1d_2/conv1d/SqueezeSqueeze7sequential/prune_low_magnitude_conv1d_2/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	*
squeeze_dims
28
6sequential/prune_low_magnitude_conv1d_2/conv1d/Squeeze
>sequential/prune_low_magnitude_conv1d_2/BiasAdd/ReadVariableOpReadVariableOpGsequential_prune_low_magnitude_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>sequential/prune_low_magnitude_conv1d_2/BiasAdd/ReadVariableOp­
/sequential/prune_low_magnitude_conv1d_2/BiasAddBiasAdd?sequential/prune_low_magnitude_conv1d_2/conv1d/Squeeze:output:0Fsequential/prune_low_magnitude_conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	21
/sequential/prune_low_magnitude_conv1d_2/BiasAddÕ
,sequential/prune_low_magnitude_conv1d_2/ReluRelu8sequential/prune_low_magnitude_conv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2.
,sequential/prune_low_magnitude_conv1d_2/Relu
0sequential/prune_low_magnitude_dropout/no_updateNoOp*
_output_shapes
 22
0sequential/prune_low_magnitude_dropout/no_update
2sequential/prune_low_magnitude_dropout/no_update_1NoOp*
_output_shapes
 24
2sequential/prune_low_magnitude_dropout/no_update_1
1sequential/prune_low_magnitude_dropout/group_depsNoOp*
_output_shapes
 23
1sequential/prune_low_magnitude_dropout/group_depsá
/sequential/prune_low_magnitude_dropout/IdentityIdentity:sequential/prune_low_magnitude_conv1d_2/Relu:activations:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	21
/sequential/prune_low_magnitude_dropout/Identity
0sequential/prune_low_magnitude_flatten/no_updateNoOp*
_output_shapes
 22
0sequential/prune_low_magnitude_flatten/no_update
2sequential/prune_low_magnitude_flatten/no_update_1NoOp*
_output_shapes
 24
2sequential/prune_low_magnitude_flatten/no_update_1
1sequential/prune_low_magnitude_flatten/group_depsNoOp*
_output_shapes
 23
1sequential/prune_low_magnitude_flatten/group_deps­
,sequential/prune_low_magnitude_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿM  2.
,sequential/prune_low_magnitude_flatten/Const
.sequential/prune_low_magnitude_flatten/ReshapeReshape8sequential/prune_low_magnitude_dropout/Identity:output:05sequential/prune_low_magnitude_flatten/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/prune_low_magnitude_flatten/Reshape~
.sequential/prune_low_magnitude_dense/no_updateNoOp*
_output_shapes
 20
.sequential/prune_low_magnitude_dense/no_update
0sequential/prune_low_magnitude_dense/no_update_1NoOp*
_output_shapes
 22
0sequential/prune_low_magnitude_dense/no_update_1õ
7sequential/prune_low_magnitude_dense/Mul/ReadVariableOpReadVariableOp@sequential_prune_low_magnitude_dense_mul_readvariableop_resource* 
_output_shapes
:
*
dtype029
7sequential/prune_low_magnitude_dense/Mul/ReadVariableOpû
9sequential/prune_low_magnitude_dense/Mul/ReadVariableOp_1ReadVariableOpBsequential_prune_low_magnitude_dense_mul_readvariableop_1_resource* 
_output_shapes
:
*
dtype02;
9sequential/prune_low_magnitude_dense/Mul/ReadVariableOp_1
(sequential/prune_low_magnitude_dense/MulMul?sequential/prune_low_magnitude_dense/Mul/ReadVariableOp:value:0Asequential/prune_low_magnitude_dense/Mul/ReadVariableOp_1:value:0*
T0* 
_output_shapes
:
2*
(sequential/prune_low_magnitude_dense/MulÏ
5sequential/prune_low_magnitude_dense/AssignVariableOpAssignVariableOp@sequential_prune_low_magnitude_dense_mul_readvariableop_resource,sequential/prune_low_magnitude_dense/Mul:z:08^sequential/prune_low_magnitude_dense/Mul/ReadVariableOp*
_output_shapes
 *
dtype027
5sequential/prune_low_magnitude_dense/AssignVariableOpæ
/sequential/prune_low_magnitude_dense/group_depsNoOp6^sequential/prune_low_magnitude_dense/AssignVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 21
/sequential/prune_low_magnitude_dense/group_depsä
1sequential/prune_low_magnitude_dense/group_deps_1NoOp0^sequential/prune_low_magnitude_dense/group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 23
1sequential/prune_low_magnitude_dense/group_deps_1³
:sequential/prune_low_magnitude_dense/MatMul/ReadVariableOpReadVariableOp@sequential_prune_low_magnitude_dense_mul_readvariableop_resource6^sequential/prune_low_magnitude_dense/AssignVariableOp* 
_output_shapes
:
*
dtype02<
:sequential/prune_low_magnitude_dense/MatMul/ReadVariableOp
+sequential/prune_low_magnitude_dense/MatMulMatMul7sequential/prune_low_magnitude_flatten/Reshape:output:0Bsequential/prune_low_magnitude_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+sequential/prune_low_magnitude_dense/MatMulû
;sequential/prune_low_magnitude_dense/BiasAdd/ReadVariableOpReadVariableOpDsequential_prune_low_magnitude_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;sequential/prune_low_magnitude_dense/BiasAdd/ReadVariableOp
,sequential/prune_low_magnitude_dense/BiasAddBiasAdd5sequential/prune_low_magnitude_dense/MatMul:product:0Csequential/prune_low_magnitude_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential/prune_low_magnitude_dense/BiasAddÐ
,sequential/prune_low_magnitude_dense/SigmoidSigmoid5sequential/prune_low_magnitude_dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential/prune_low_magnitude_dense/Sigmoidë
IdentityIdentity0sequential/prune_low_magnitude_dense/Sigmoid:y:07^sequential/prune_low_magnitude_conv1d/AssignVariableOp9^sequential/prune_low_magnitude_conv1d_1/AssignVariableOp9^sequential/prune_low_magnitude_conv1d_2/AssignVariableOp6^sequential/prune_low_magnitude_dense/AssignVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿâ	::::::::::::2p
6sequential/prune_low_magnitude_conv1d/AssignVariableOp6sequential/prune_low_magnitude_conv1d/AssignVariableOp2t
8sequential/prune_low_magnitude_conv1d_1/AssignVariableOp8sequential/prune_low_magnitude_conv1d_1/AssignVariableOp2t
8sequential/prune_low_magnitude_conv1d_2/AssignVariableOp8sequential/prune_low_magnitude_conv1d_2/AssignVariableOp2n
5sequential/prune_low_magnitude_dense/AssignVariableOp5sequential/prune_low_magnitude_dense/AssignVariableOp:Z V
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	
&
_user_specified_nameconv1d_input:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ö
Ê
Pprune_low_magnitude_conv1d_2_assert_greater_equal_Assert_AssertGuard_true_317752B
>identity_prune_low_magnitude_conv1d_2_assert_greater_equal_all

placeholder	
placeholder_1	

identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOp
IdentityIdentity>identity_prune_low_magnitude_conv1d_2_assert_greater_equal_all^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ð
Æ
Nprune_low_magnitude_conv1d_assert_greater_equal_Assert_AssertGuard_true_317420@
<identity_prune_low_magnitude_conv1d_assert_greater_equal_all

placeholder	
placeholder_1	

identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOp
IdentityIdentity<identity_prune_low_magnitude_conv1d_assert_greater_equal_all^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

=
cond_true_319106
identity_logicaland_1


identity_1
6

group_depsNoOp*
_output_shapes
 2

group_depse
IdentityIdentityidentity_logicaland_1^group_deps*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : 

_output_shapes
: 
ó

<__inference_prune_low_magnitude_flatten_layer_call_fn_319245

inputs
unknown
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*`
f[RY
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_3169212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÙ	:22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	
 
_user_specified_nameinputs:

_output_shapes
: 
ùg

__inference__traced_save_319624
file_prefix>
:savev2_prune_low_magnitude_conv1d_mask_read_readvariableopC
?savev2_prune_low_magnitude_conv1d_threshold_read_readvariableopF
Bsavev2_prune_low_magnitude_conv1d_pruning_step_read_readvariableop	@
<savev2_prune_low_magnitude_conv1d_1_mask_read_readvariableopE
Asavev2_prune_low_magnitude_conv1d_1_threshold_read_readvariableopH
Dsavev2_prune_low_magnitude_conv1d_1_pruning_step_read_readvariableop	@
<savev2_prune_low_magnitude_conv1d_2_mask_read_readvariableopE
Asavev2_prune_low_magnitude_conv1d_2_threshold_read_readvariableopH
Dsavev2_prune_low_magnitude_conv1d_2_pruning_step_read_readvariableop	G
Csavev2_prune_low_magnitude_dropout_pruning_step_read_readvariableop	G
Csavev2_prune_low_magnitude_flatten_pruning_step_read_readvariableop	=
9savev2_prune_low_magnitude_dense_mask_read_readvariableopB
>savev2_prune_low_magnitude_dense_threshold_read_readvariableopE
Asavev2_prune_low_magnitude_dense_pruning_step_read_readvariableop	#
savev2_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_conv1d_kernel_m_read_readvariableop1
-savev2_adam_conv1d_bias_m_read_readvariableop5
1savev2_adam_conv1d_1_kernel_m_read_readvariableop3
/savev2_adam_conv1d_1_bias_m_read_readvariableop5
1savev2_adam_conv1d_2_kernel_m_read_readvariableop3
/savev2_adam_conv1d_2_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop3
/savev2_adam_conv1d_kernel_v_read_readvariableop1
-savev2_adam_conv1d_bias_v_read_readvariableop5
1savev2_adam_conv1d_1_kernel_v_read_readvariableop3
/savev2_adam_conv1d_1_bias_v_read_readvariableop5
1savev2_adam_conv1d_2_kernel_v_read_readvariableop3
/savev2_adam_conv1d_2_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_1_const

identity_1¢MergeV2Checkpoints¢SaveV2¢SaveV2_1
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_1eac8de4e3d9433ca72890a70f47ab63/part2	
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
value	B :2

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
ShardedFilenameè
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*ú
valueðBí/B4layer_with_weights-0/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-0/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-1/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-1/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-2/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-2/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-3/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-4/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-5/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-5/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesæ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0:savev2_prune_low_magnitude_conv1d_mask_read_readvariableop?savev2_prune_low_magnitude_conv1d_threshold_read_readvariableopBsavev2_prune_low_magnitude_conv1d_pruning_step_read_readvariableop<savev2_prune_low_magnitude_conv1d_1_mask_read_readvariableopAsavev2_prune_low_magnitude_conv1d_1_threshold_read_readvariableopDsavev2_prune_low_magnitude_conv1d_1_pruning_step_read_readvariableop<savev2_prune_low_magnitude_conv1d_2_mask_read_readvariableopAsavev2_prune_low_magnitude_conv1d_2_threshold_read_readvariableopDsavev2_prune_low_magnitude_conv1d_2_pruning_step_read_readvariableopCsavev2_prune_low_magnitude_dropout_pruning_step_read_readvariableopCsavev2_prune_low_magnitude_flatten_pruning_step_read_readvariableop9savev2_prune_low_magnitude_dense_mask_read_readvariableop>savev2_prune_low_magnitude_dense_threshold_read_readvariableopAsavev2_prune_low_magnitude_dense_pruning_step_read_readvariableopsavev2_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_conv1d_kernel_m_read_readvariableop-savev2_adam_conv1d_bias_m_read_readvariableop1savev2_adam_conv1d_1_kernel_m_read_readvariableop/savev2_adam_conv1d_1_bias_m_read_readvariableop1savev2_adam_conv1d_2_kernel_m_read_readvariableop/savev2_adam_conv1d_2_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop/savev2_adam_conv1d_kernel_v_read_readvariableop-savev2_adam_conv1d_bias_v_read_readvariableop1savev2_adam_conv1d_1_kernel_v_read_readvariableop/savev2_adam_conv1d_1_bias_v_read_readvariableop1savev2_adam_conv1d_2_kernel_v_read_readvariableop/savev2_adam_conv1d_2_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *=
dtypes3
12/							2
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard¬
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1¢
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesÏ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1ã
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¬
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ß
_input_shapesÍ
Ê: :: : :: : :: : : : :
: : : : : : : :::::::
:: : : : :::::::
::::::::
:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
::

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
::

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
::

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::&"
 
_output_shapes
:
: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :( $
"
_output_shapes
:: !

_output_shapes
::("$
"
_output_shapes
:: #

_output_shapes
::($$
"
_output_shapes
:: %

_output_shapes
::&&"
 
_output_shapes
:
: '

_output_shapes
::(($
"
_output_shapes
:: )

_output_shapes
::(*$
"
_output_shapes
:: +

_output_shapes
::(,$
"
_output_shapes
:: -

_output_shapes
::&."
 
_output_shapes
:
: /

_output_shapes
::0

_output_shapes
: 
é
²
Nprune_low_magnitude_dense_assert_greater_equal_Assert_AssertGuard_false_318093=
9assert_prune_low_magnitude_dense_assert_greater_equal_all
H
Dassert_prune_low_magnitude_dense_assert_greater_equal_readvariableop	;
7assert_prune_low_magnitude_dense_assert_greater_equal_y	

identity_1
¢Assertî
Assert/data_0Const*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2
Assert/data_0
Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2
Assert/data_1¥
Assert/data_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (prune_low_magnitude_dense/assert_greater_equal/ReadVariableOp:0) = 2
Assert/data_2
Assert/data_4Const*
_output_shapes
: *
dtype0*J
valueAB? B9y (prune_low_magnitude_dense/assert_greater_equal/y:0) = 2
Assert/data_4Û
AssertAssert9assert_prune_low_magnitude_dense_assert_greater_equal_allAssert/data_0:output:0Assert/data_1:output:0Assert/data_2:output:0Dassert_prune_low_magnitude_dense_assert_greater_equal_readvariableopAssert/data_4:output:07assert_prune_low_magnitude_dense_assert_greater_equal_y*
T

2		*
_output_shapes
 2
Assert
IdentityIdentity9assert_prune_low_magnitude_dense_assert_greater_equal_all^Assert*
T0
*
_output_shapes
: 2

Identitya

Identity_1IdentityIdentity:output:0^Assert*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : 2
AssertAssert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
 

F__inference_sequential_layer_call_and_return_conditional_losses_318307

inputs:
6prune_low_magnitude_conv1d_mul_readvariableop_resource<
8prune_low_magnitude_conv1d_mul_readvariableop_1_resource>
:prune_low_magnitude_conv1d_biasadd_readvariableop_resource<
8prune_low_magnitude_conv1d_1_mul_readvariableop_resource>
:prune_low_magnitude_conv1d_1_mul_readvariableop_1_resource@
<prune_low_magnitude_conv1d_1_biasadd_readvariableop_resource<
8prune_low_magnitude_conv1d_2_mul_readvariableop_resource>
:prune_low_magnitude_conv1d_2_mul_readvariableop_1_resource@
<prune_low_magnitude_conv1d_2_biasadd_readvariableop_resource9
5prune_low_magnitude_dense_mul_readvariableop_resource;
7prune_low_magnitude_dense_mul_readvariableop_1_resource=
9prune_low_magnitude_dense_biasadd_readvariableop_resource
identity¢+prune_low_magnitude_conv1d/AssignVariableOp¢-prune_low_magnitude_conv1d_1/AssignVariableOp¢-prune_low_magnitude_conv1d_2/AssignVariableOp¢*prune_low_magnitude_dense/AssignVariableOpj
$prune_low_magnitude_conv1d/no_updateNoOp*
_output_shapes
 2&
$prune_low_magnitude_conv1d/no_updaten
&prune_low_magnitude_conv1d/no_update_1NoOp*
_output_shapes
 2(
&prune_low_magnitude_conv1d/no_update_1Ù
-prune_low_magnitude_conv1d/Mul/ReadVariableOpReadVariableOp6prune_low_magnitude_conv1d_mul_readvariableop_resource*"
_output_shapes
:*
dtype02/
-prune_low_magnitude_conv1d/Mul/ReadVariableOpß
/prune_low_magnitude_conv1d/Mul/ReadVariableOp_1ReadVariableOp8prune_low_magnitude_conv1d_mul_readvariableop_1_resource*"
_output_shapes
:*
dtype021
/prune_low_magnitude_conv1d/Mul/ReadVariableOp_1ä
prune_low_magnitude_conv1d/MulMul5prune_low_magnitude_conv1d/Mul/ReadVariableOp:value:07prune_low_magnitude_conv1d/Mul/ReadVariableOp_1:value:0*
T0*"
_output_shapes
:2 
prune_low_magnitude_conv1d/Mul
+prune_low_magnitude_conv1d/AssignVariableOpAssignVariableOp6prune_low_magnitude_conv1d_mul_readvariableop_resource"prune_low_magnitude_conv1d/Mul:z:0.^prune_low_magnitude_conv1d/Mul/ReadVariableOp*
_output_shapes
 *
dtype02-
+prune_low_magnitude_conv1d/AssignVariableOpÈ
%prune_low_magnitude_conv1d/group_depsNoOp,^prune_low_magnitude_conv1d/AssignVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2'
%prune_low_magnitude_conv1d/group_depsÆ
'prune_low_magnitude_conv1d/group_deps_1NoOp&^prune_low_magnitude_conv1d/group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2)
'prune_low_magnitude_conv1d/group_deps_1¦
0prune_low_magnitude_conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :22
0prune_low_magnitude_conv1d/conv1d/ExpandDims/dimè
,prune_low_magnitude_conv1d/conv1d/ExpandDims
ExpandDimsinputs9prune_low_magnitude_conv1d/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	2.
,prune_low_magnitude_conv1d/conv1d/ExpandDims§
=prune_low_magnitude_conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6prune_low_magnitude_conv1d_mul_readvariableop_resource,^prune_low_magnitude_conv1d/AssignVariableOp*"
_output_shapes
:*
dtype02?
=prune_low_magnitude_conv1d/conv1d/ExpandDims_1/ReadVariableOpª
2prune_low_magnitude_conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 24
2prune_low_magnitude_conv1d/conv1d/ExpandDims_1/dim£
.prune_low_magnitude_conv1d/conv1d/ExpandDims_1
ExpandDimsEprune_low_magnitude_conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0;prune_low_magnitude_conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:20
.prune_low_magnitude_conv1d/conv1d/ExpandDims_1¤
!prune_low_magnitude_conv1d/conv1dConv2D5prune_low_magnitude_conv1d/conv1d/ExpandDims:output:07prune_low_magnitude_conv1d/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	*
paddingVALID*
strides
2#
!prune_low_magnitude_conv1d/conv1dÛ
)prune_low_magnitude_conv1d/conv1d/SqueezeSqueeze*prune_low_magnitude_conv1d/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	*
squeeze_dims
2+
)prune_low_magnitude_conv1d/conv1d/SqueezeÝ
1prune_low_magnitude_conv1d/BiasAdd/ReadVariableOpReadVariableOp:prune_low_magnitude_conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1prune_low_magnitude_conv1d/BiasAdd/ReadVariableOpù
"prune_low_magnitude_conv1d/BiasAddBiasAdd2prune_low_magnitude_conv1d/conv1d/Squeeze:output:09prune_low_magnitude_conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	2$
"prune_low_magnitude_conv1d/BiasAdd®
prune_low_magnitude_conv1d/ReluRelu+prune_low_magnitude_conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	2!
prune_low_magnitude_conv1d/Relun
&prune_low_magnitude_conv1d_1/no_updateNoOp*
_output_shapes
 2(
&prune_low_magnitude_conv1d_1/no_updater
(prune_low_magnitude_conv1d_1/no_update_1NoOp*
_output_shapes
 2*
(prune_low_magnitude_conv1d_1/no_update_1ß
/prune_low_magnitude_conv1d_1/Mul/ReadVariableOpReadVariableOp8prune_low_magnitude_conv1d_1_mul_readvariableop_resource*"
_output_shapes
:*
dtype021
/prune_low_magnitude_conv1d_1/Mul/ReadVariableOpå
1prune_low_magnitude_conv1d_1/Mul/ReadVariableOp_1ReadVariableOp:prune_low_magnitude_conv1d_1_mul_readvariableop_1_resource*"
_output_shapes
:*
dtype023
1prune_low_magnitude_conv1d_1/Mul/ReadVariableOp_1ì
 prune_low_magnitude_conv1d_1/MulMul7prune_low_magnitude_conv1d_1/Mul/ReadVariableOp:value:09prune_low_magnitude_conv1d_1/Mul/ReadVariableOp_1:value:0*
T0*"
_output_shapes
:2"
 prune_low_magnitude_conv1d_1/Mul§
-prune_low_magnitude_conv1d_1/AssignVariableOpAssignVariableOp8prune_low_magnitude_conv1d_1_mul_readvariableop_resource$prune_low_magnitude_conv1d_1/Mul:z:00^prune_low_magnitude_conv1d_1/Mul/ReadVariableOp*
_output_shapes
 *
dtype02/
-prune_low_magnitude_conv1d_1/AssignVariableOpÎ
'prune_low_magnitude_conv1d_1/group_depsNoOp.^prune_low_magnitude_conv1d_1/AssignVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2)
'prune_low_magnitude_conv1d_1/group_depsÌ
)prune_low_magnitude_conv1d_1/group_deps_1NoOp(^prune_low_magnitude_conv1d_1/group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2+
)prune_low_magnitude_conv1d_1/group_deps_1ª
2prune_low_magnitude_conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2prune_low_magnitude_conv1d_1/conv1d/ExpandDims/dim
.prune_low_magnitude_conv1d_1/conv1d/ExpandDims
ExpandDims-prune_low_magnitude_conv1d/Relu:activations:0;prune_low_magnitude_conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	20
.prune_low_magnitude_conv1d_1/conv1d/ExpandDims¯
?prune_low_magnitude_conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8prune_low_magnitude_conv1d_1_mul_readvariableop_resource.^prune_low_magnitude_conv1d_1/AssignVariableOp*"
_output_shapes
:*
dtype02A
?prune_low_magnitude_conv1d_1/conv1d/ExpandDims_1/ReadVariableOp®
4prune_low_magnitude_conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4prune_low_magnitude_conv1d_1/conv1d/ExpandDims_1/dim«
0prune_low_magnitude_conv1d_1/conv1d/ExpandDims_1
ExpandDimsGprune_low_magnitude_conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0=prune_low_magnitude_conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0prune_low_magnitude_conv1d_1/conv1d/ExpandDims_1¬
#prune_low_magnitude_conv1d_1/conv1dConv2D7prune_low_magnitude_conv1d_1/conv1d/ExpandDims:output:09prune_low_magnitude_conv1d_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	*
paddingVALID*
strides
2%
#prune_low_magnitude_conv1d_1/conv1dá
+prune_low_magnitude_conv1d_1/conv1d/SqueezeSqueeze,prune_low_magnitude_conv1d_1/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	*
squeeze_dims
2-
+prune_low_magnitude_conv1d_1/conv1d/Squeezeã
3prune_low_magnitude_conv1d_1/BiasAdd/ReadVariableOpReadVariableOp<prune_low_magnitude_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3prune_low_magnitude_conv1d_1/BiasAdd/ReadVariableOp
$prune_low_magnitude_conv1d_1/BiasAddBiasAdd4prune_low_magnitude_conv1d_1/conv1d/Squeeze:output:0;prune_low_magnitude_conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	2&
$prune_low_magnitude_conv1d_1/BiasAdd´
!prune_low_magnitude_conv1d_1/ReluRelu-prune_low_magnitude_conv1d_1/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	2#
!prune_low_magnitude_conv1d_1/Relun
&prune_low_magnitude_conv1d_2/no_updateNoOp*
_output_shapes
 2(
&prune_low_magnitude_conv1d_2/no_updater
(prune_low_magnitude_conv1d_2/no_update_1NoOp*
_output_shapes
 2*
(prune_low_magnitude_conv1d_2/no_update_1ß
/prune_low_magnitude_conv1d_2/Mul/ReadVariableOpReadVariableOp8prune_low_magnitude_conv1d_2_mul_readvariableop_resource*"
_output_shapes
:*
dtype021
/prune_low_magnitude_conv1d_2/Mul/ReadVariableOpå
1prune_low_magnitude_conv1d_2/Mul/ReadVariableOp_1ReadVariableOp:prune_low_magnitude_conv1d_2_mul_readvariableop_1_resource*"
_output_shapes
:*
dtype023
1prune_low_magnitude_conv1d_2/Mul/ReadVariableOp_1ì
 prune_low_magnitude_conv1d_2/MulMul7prune_low_magnitude_conv1d_2/Mul/ReadVariableOp:value:09prune_low_magnitude_conv1d_2/Mul/ReadVariableOp_1:value:0*
T0*"
_output_shapes
:2"
 prune_low_magnitude_conv1d_2/Mul§
-prune_low_magnitude_conv1d_2/AssignVariableOpAssignVariableOp8prune_low_magnitude_conv1d_2_mul_readvariableop_resource$prune_low_magnitude_conv1d_2/Mul:z:00^prune_low_magnitude_conv1d_2/Mul/ReadVariableOp*
_output_shapes
 *
dtype02/
-prune_low_magnitude_conv1d_2/AssignVariableOpÎ
'prune_low_magnitude_conv1d_2/group_depsNoOp.^prune_low_magnitude_conv1d_2/AssignVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2)
'prune_low_magnitude_conv1d_2/group_depsÌ
)prune_low_magnitude_conv1d_2/group_deps_1NoOp(^prune_low_magnitude_conv1d_2/group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2+
)prune_low_magnitude_conv1d_2/group_deps_1ª
2prune_low_magnitude_conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2prune_low_magnitude_conv1d_2/conv1d/ExpandDims/dim
.prune_low_magnitude_conv1d_2/conv1d/ExpandDims
ExpandDims/prune_low_magnitude_conv1d_1/Relu:activations:0;prune_low_magnitude_conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	20
.prune_low_magnitude_conv1d_2/conv1d/ExpandDims¯
?prune_low_magnitude_conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8prune_low_magnitude_conv1d_2_mul_readvariableop_resource.^prune_low_magnitude_conv1d_2/AssignVariableOp*"
_output_shapes
:*
dtype02A
?prune_low_magnitude_conv1d_2/conv1d/ExpandDims_1/ReadVariableOp®
4prune_low_magnitude_conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4prune_low_magnitude_conv1d_2/conv1d/ExpandDims_1/dim«
0prune_low_magnitude_conv1d_2/conv1d/ExpandDims_1
ExpandDimsGprune_low_magnitude_conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0=prune_low_magnitude_conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0prune_low_magnitude_conv1d_2/conv1d/ExpandDims_1¬
#prune_low_magnitude_conv1d_2/conv1dConv2D7prune_low_magnitude_conv1d_2/conv1d/ExpandDims:output:09prune_low_magnitude_conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	*
paddingVALID*
strides
2%
#prune_low_magnitude_conv1d_2/conv1dá
+prune_low_magnitude_conv1d_2/conv1d/SqueezeSqueeze,prune_low_magnitude_conv1d_2/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	*
squeeze_dims
2-
+prune_low_magnitude_conv1d_2/conv1d/Squeezeã
3prune_low_magnitude_conv1d_2/BiasAdd/ReadVariableOpReadVariableOp<prune_low_magnitude_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3prune_low_magnitude_conv1d_2/BiasAdd/ReadVariableOp
$prune_low_magnitude_conv1d_2/BiasAddBiasAdd4prune_low_magnitude_conv1d_2/conv1d/Squeeze:output:0;prune_low_magnitude_conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2&
$prune_low_magnitude_conv1d_2/BiasAdd´
!prune_low_magnitude_conv1d_2/ReluRelu-prune_low_magnitude_conv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2#
!prune_low_magnitude_conv1d_2/Relul
%prune_low_magnitude_dropout/no_updateNoOp*
_output_shapes
 2'
%prune_low_magnitude_dropout/no_updatep
'prune_low_magnitude_dropout/no_update_1NoOp*
_output_shapes
 2)
'prune_low_magnitude_dropout/no_update_1n
&prune_low_magnitude_dropout/group_depsNoOp*
_output_shapes
 2(
&prune_low_magnitude_dropout/group_depsÀ
$prune_low_magnitude_dropout/IdentityIdentity/prune_low_magnitude_conv1d_2/Relu:activations:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2&
$prune_low_magnitude_dropout/Identityl
%prune_low_magnitude_flatten/no_updateNoOp*
_output_shapes
 2'
%prune_low_magnitude_flatten/no_updatep
'prune_low_magnitude_flatten/no_update_1NoOp*
_output_shapes
 2)
'prune_low_magnitude_flatten/no_update_1n
&prune_low_magnitude_flatten/group_depsNoOp*
_output_shapes
 2(
&prune_low_magnitude_flatten/group_deps
!prune_low_magnitude_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿM  2#
!prune_low_magnitude_flatten/Constä
#prune_low_magnitude_flatten/ReshapeReshape-prune_low_magnitude_dropout/Identity:output:0*prune_low_magnitude_flatten/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#prune_low_magnitude_flatten/Reshapeh
#prune_low_magnitude_dense/no_updateNoOp*
_output_shapes
 2%
#prune_low_magnitude_dense/no_updatel
%prune_low_magnitude_dense/no_update_1NoOp*
_output_shapes
 2'
%prune_low_magnitude_dense/no_update_1Ô
,prune_low_magnitude_dense/Mul/ReadVariableOpReadVariableOp5prune_low_magnitude_dense_mul_readvariableop_resource* 
_output_shapes
:
*
dtype02.
,prune_low_magnitude_dense/Mul/ReadVariableOpÚ
.prune_low_magnitude_dense/Mul/ReadVariableOp_1ReadVariableOp7prune_low_magnitude_dense_mul_readvariableop_1_resource* 
_output_shapes
:
*
dtype020
.prune_low_magnitude_dense/Mul/ReadVariableOp_1Þ
prune_low_magnitude_dense/MulMul4prune_low_magnitude_dense/Mul/ReadVariableOp:value:06prune_low_magnitude_dense/Mul/ReadVariableOp_1:value:0*
T0* 
_output_shapes
:
2
prune_low_magnitude_dense/Mul
*prune_low_magnitude_dense/AssignVariableOpAssignVariableOp5prune_low_magnitude_dense_mul_readvariableop_resource!prune_low_magnitude_dense/Mul:z:0-^prune_low_magnitude_dense/Mul/ReadVariableOp*
_output_shapes
 *
dtype02,
*prune_low_magnitude_dense/AssignVariableOpÅ
$prune_low_magnitude_dense/group_depsNoOp+^prune_low_magnitude_dense/AssignVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2&
$prune_low_magnitude_dense/group_depsÃ
&prune_low_magnitude_dense/group_deps_1NoOp%^prune_low_magnitude_dense/group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2(
&prune_low_magnitude_dense/group_deps_1
/prune_low_magnitude_dense/MatMul/ReadVariableOpReadVariableOp5prune_low_magnitude_dense_mul_readvariableop_resource+^prune_low_magnitude_dense/AssignVariableOp* 
_output_shapes
:
*
dtype021
/prune_low_magnitude_dense/MatMul/ReadVariableOpç
 prune_low_magnitude_dense/MatMulMatMul,prune_low_magnitude_flatten/Reshape:output:07prune_low_magnitude_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 prune_low_magnitude_dense/MatMulÚ
0prune_low_magnitude_dense/BiasAdd/ReadVariableOpReadVariableOp9prune_low_magnitude_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0prune_low_magnitude_dense/BiasAdd/ReadVariableOpé
!prune_low_magnitude_dense/BiasAddBiasAdd*prune_low_magnitude_dense/MatMul:product:08prune_low_magnitude_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!prune_low_magnitude_dense/BiasAdd¯
!prune_low_magnitude_dense/SigmoidSigmoid*prune_low_magnitude_dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!prune_low_magnitude_dense/Sigmoid´
IdentityIdentity%prune_low_magnitude_dense/Sigmoid:y:0,^prune_low_magnitude_conv1d/AssignVariableOp.^prune_low_magnitude_conv1d_1/AssignVariableOp.^prune_low_magnitude_conv1d_2/AssignVariableOp+^prune_low_magnitude_dense/AssignVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿâ	::::::::::::2Z
+prune_low_magnitude_conv1d/AssignVariableOp+prune_low_magnitude_conv1d/AssignVariableOp2^
-prune_low_magnitude_conv1d_1/AssignVariableOp-prune_low_magnitude_conv1d_1/AssignVariableOp2^
-prune_low_magnitude_conv1d_2/AssignVariableOp-prune_low_magnitude_conv1d_2/AssignVariableOp2X
*prune_low_magnitude_dense/AssignVariableOp*prune_low_magnitude_dense/AssignVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	
 
_user_specified_nameinputs:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ò

cond_false_316104
placeholder
placeholder_1
placeholder_2
placeholder_3
identity_logicaland_1


identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOp_
IdentityIdentityidentity_logicaland_1^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
::::: : 

_output_shapes
: :
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
: 
ÿ
>
cond_false_319218
identity_logicaland_1


identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOp_
IdentityIdentityidentity_logicaland_1^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : 

_output_shapes
: 
éZ
Ð
W__inference_prune_low_magnitude_dropout_layer_call_and_return_conditional_losses_316811

inputs
readvariableop_resource
identity¢AssignVariableOp¢'assert_greater_equal/Assert/AssertGuardp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOpP
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
add/y\
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: 2
add
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOpA
updateNoOp^AssignVariableOp*
_output_shapes
 2
update­
#assert_greater_equal/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
: *
dtype0	2%
#assert_greater_equal/ReadVariableOpr
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
assert_greater_equal/yÅ
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2#
!assert_greater_equal/GreaterEqual{
assert_greater_equal/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
assert_greater_equal/Const
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/Const:output:0*
_output_shapes
: 2
assert_greater_equal/All
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2#
!assert_greater_equal/Assert/Const¶
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2%
#assert_greater_equal/Assert/Const_1·
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2%
#assert_greater_equal/Assert/Const_2ª
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2%
#assert_greater_equal/Assert/Const_3
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_316734*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_3167332)
'assert_greater_equal/Assert/AssertGuardÃ
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identityú
0polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	22
0polynomial_decay_pruning_schedule/ReadVariableOpÇ
'polynomial_decay_pruning_schedule/sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'polynomial_decay_pruning_schedule/sub/yâ
%polynomial_decay_pruning_schedule/subSub8polynomial_decay_pruning_schedule/ReadVariableOp:value:00polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/sub³
&polynomial_decay_pruning_schedule/CastCast)polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2(
&polynomial_decay_pruning_schedule/CastÒ
+polynomial_decay_pruning_schedule/truediv/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 * ¨G2-
+polynomial_decay_pruning_schedule/truediv/yä
)polynomial_decay_pruning_schedule/truedivRealDiv*polynomial_decay_pruning_schedule/Cast:y:04polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/truedivÒ
+polynomial_decay_pruning_schedule/Maximum/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *    2-
+polynomial_decay_pruning_schedule/Maximum/xç
)polynomial_decay_pruning_schedule/MaximumMaximum4polynomial_decay_pruning_schedule/Maximum/x:output:0-polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/MaximumÒ
+polynomial_decay_pruning_schedule/Minimum/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+polynomial_decay_pruning_schedule/Minimum/xç
)polynomial_decay_pruning_schedule/MinimumMinimum4polynomial_decay_pruning_schedule/Minimum/x:output:0-polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/MinimumÎ
)polynomial_decay_pruning_schedule/sub_1/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)polynomial_decay_pruning_schedule/sub_1/xÝ
'polynomial_decay_pruning_schedule/sub_1Sub2polynomial_decay_pruning_schedule/sub_1/x:output:0-polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2)
'polynomial_decay_pruning_schedule/sub_1Ê
'polynomial_decay_pruning_schedule/Pow/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  @@2)
'polynomial_decay_pruning_schedule/Pow/yÕ
%polynomial_decay_pruning_schedule/PowPow+polynomial_decay_pruning_schedule/sub_1:z:00polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/PowÊ
'polynomial_decay_pruning_schedule/Mul/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *¾2)
'polynomial_decay_pruning_schedule/Mul/xÓ
%polynomial_decay_pruning_schedule/MulMul0polynomial_decay_pruning_schedule/Mul/x:output:0)polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/MulÔ
,polynomial_decay_pruning_schedule/sparsity/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2.
,polynomial_decay_pruning_schedule/sparsity/yâ
*polynomial_decay_pruning_schedule/sparsityAdd)polynomial_decay_pruning_schedule/Mul:z:05polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2,
*polynomial_decay_pruning_schedule/sparsityÐ
GreaterEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOp
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqualÊ
LessEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqual
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
Less/x
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0	*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd¾
Sub/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOp
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
Sub

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorMod
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1¦
condStatelessIfLogicalAnd_1:z:0LogicalAnd_1:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *$
else_branchR
cond_false_316791*
output_shapes
: *#
then_branchR
cond_true_3167902
condZ
cond/IdentityIdentitycond:output:0*
T0
*
_output_shapes
: 2
cond/Identityu
update_1NoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 2

update_16

group_depsNoOp*
_output_shapes
 2

group_depsc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/GreaterEqual/yÃ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2
dropout/Mul_1§
IdentityIdentitydropout/Mul_1:z:0^AssignVariableOp(^assert_greater_equal/Assert/AssertGuard*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÙ	:2$
AssignVariableOpAssignVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	
 
_user_specified_nameinputs:

_output_shapes
: 
ëK
Ê
+prune_low_magnitude_conv1d_cond_true_317477=
9polynomial_decay_pruning_schedule_readvariableop_resource+
'pruning_ops_abs_readvariableop_resource
assignvariableop_resource
assignvariableop_1_resource4
0identity_prune_low_magnitude_conv1d_logicaland_1


identity_1
¢AssignVariableOp¢AssignVariableOp_1Ö
0polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	22
0polynomial_decay_pruning_schedule/ReadVariableOp
'polynomial_decay_pruning_schedule/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'polynomial_decay_pruning_schedule/sub/yâ
%polynomial_decay_pruning_schedule/subSub8polynomial_decay_pruning_schedule/ReadVariableOp:value:00polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/sub³
&polynomial_decay_pruning_schedule/CastCast)polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2(
&polynomial_decay_pruning_schedule/Cast
+polynomial_decay_pruning_schedule/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ¨G2-
+polynomial_decay_pruning_schedule/truediv/yä
)polynomial_decay_pruning_schedule/truedivRealDiv*polynomial_decay_pruning_schedule/Cast:y:04polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/truediv
+polynomial_decay_pruning_schedule/Maximum/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+polynomial_decay_pruning_schedule/Maximum/xç
)polynomial_decay_pruning_schedule/MaximumMaximum4polynomial_decay_pruning_schedule/Maximum/x:output:0-polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/Maximum
+polynomial_decay_pruning_schedule/Minimum/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+polynomial_decay_pruning_schedule/Minimum/xç
)polynomial_decay_pruning_schedule/MinimumMinimum4polynomial_decay_pruning_schedule/Minimum/x:output:0-polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/Minimum
)polynomial_decay_pruning_schedule/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)polynomial_decay_pruning_schedule/sub_1/xÝ
'polynomial_decay_pruning_schedule/sub_1Sub2polynomial_decay_pruning_schedule/sub_1/x:output:0-polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2)
'polynomial_decay_pruning_schedule/sub_1
'polynomial_decay_pruning_schedule/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2)
'polynomial_decay_pruning_schedule/Pow/yÕ
%polynomial_decay_pruning_schedule/PowPow+polynomial_decay_pruning_schedule/sub_1:z:00polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/Pow
'polynomial_decay_pruning_schedule/Mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¾2)
'polynomial_decay_pruning_schedule/Mul/xÓ
%polynomial_decay_pruning_schedule/MulMul0polynomial_decay_pruning_schedule/Mul/x:output:0)polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/Mul¡
,polynomial_decay_pruning_schedule/sparsity/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2.
,polynomial_decay_pruning_schedule/sparsity/yâ
*polynomial_decay_pruning_schedule/sparsityAdd)polynomial_decay_pruning_schedule/Mul:z:05polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2,
*polynomial_decay_pruning_schedule/sparsity¬
GreaterEqual/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOpb
GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqual¦
LessEqual/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp^
LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqualT
Less/xConst*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
Less/xR
Less/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0	*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd
Sub/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOpP
Sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
SubZ

FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorModT
Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1¬
pruning_ops/Abs/ReadVariableOpReadVariableOp'pruning_ops_abs_readvariableop_resource*"
_output_shapes
:*
dtype02 
pruning_ops/Abs/ReadVariableOp~
pruning_ops/AbsAbs&pruning_ops/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:2
pruning_ops/Absf
pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value	B :02
pruning_ops/Sizew
pruning_ops/CastCastpruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
pruning_ops/Castk
pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
pruning_ops/sub/x
pruning_ops/subSubpruning_ops/sub/x:output:0.polynomial_decay_pruning_schedule/sparsity:z:0*
T0*
_output_shapes
: 2
pruning_ops/subu
pruning_ops/mulMulpruning_ops/Cast:y:0pruning_ops/sub:z:0*
T0*
_output_shapes
: 2
pruning_ops/mule
pruning_ops/RoundRoundpruning_ops/mul:z:0*
T0*
_output_shapes
: 2
pruning_ops/Rounds
pruning_ops/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
pruning_ops/Maximum/y
pruning_ops/MaximumMaximumpruning_ops/Round:y:0pruning_ops/Maximum/y:output:0*
T0*
_output_shapes
: 2
pruning_ops/Maximumy
pruning_ops/Cast_1Castpruning_ops/Maximum:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
pruning_ops/Cast_1
pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
pruning_ops/Reshape/shape
pruning_ops/ReshapeReshapepruning_ops/Abs:y:0"pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes
:02
pruning_ops/Reshapej
pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value	B :02
pruning_ops/Size_1
pruning_ops/TopKV2TopKV2pruning_ops/Reshape:output:0pruning_ops/Size_1:output:0*
T0* 
_output_shapes
:0:02
pruning_ops/TopKV2l
pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
pruning_ops/sub_1/y
pruning_ops/sub_1Subpruning_ops/Cast_1:y:0pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: 2
pruning_ops/sub_1x
pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
pruning_ops/GatherV2/axisÔ
pruning_ops/GatherV2GatherV2pruning_ops/TopKV2:values:0pruning_ops/sub_1:z:0"pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: 2
pruning_ops/GatherV2¥
pruning_ops/GreaterEqualGreaterEqualpruning_ops/Abs:y:0pruning_ops/GatherV2:output:0*
T0*"
_output_shapes
:2
pruning_ops/GreaterEqual
pruning_ops/Cast_2Castpruning_ops/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
:2
pruning_ops/Cast_2
AssignVariableOpAssignVariableOpassignvariableop_resourcepruning_ops/Cast_2:y:0*
_output_shapes
 *
dtype02
AssignVariableOp
AssignVariableOp_1AssignVariableOpassignvariableop_1_resourcepruning_ops/GatherV2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1

group_depsNoOp^AssignVariableOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_deps
IdentityIdentity0identity_prune_low_magnitude_conv1d_logicaland_1^group_deps*
T0
*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^AssignVariableOp^AssignVariableOp_1*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
::::: 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_1: 

_output_shapes
: :
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
: 
öK
Î
-prune_low_magnitude_conv1d_1_cond_true_317643=
9polynomial_decay_pruning_schedule_readvariableop_resource+
'pruning_ops_abs_readvariableop_resource
assignvariableop_resource
assignvariableop_1_resource6
2identity_prune_low_magnitude_conv1d_1_logicaland_1


identity_1
¢AssignVariableOp¢AssignVariableOp_1Ö
0polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	22
0polynomial_decay_pruning_schedule/ReadVariableOp
'polynomial_decay_pruning_schedule/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'polynomial_decay_pruning_schedule/sub/yâ
%polynomial_decay_pruning_schedule/subSub8polynomial_decay_pruning_schedule/ReadVariableOp:value:00polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/sub³
&polynomial_decay_pruning_schedule/CastCast)polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2(
&polynomial_decay_pruning_schedule/Cast
+polynomial_decay_pruning_schedule/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ¨G2-
+polynomial_decay_pruning_schedule/truediv/yä
)polynomial_decay_pruning_schedule/truedivRealDiv*polynomial_decay_pruning_schedule/Cast:y:04polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/truediv
+polynomial_decay_pruning_schedule/Maximum/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+polynomial_decay_pruning_schedule/Maximum/xç
)polynomial_decay_pruning_schedule/MaximumMaximum4polynomial_decay_pruning_schedule/Maximum/x:output:0-polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/Maximum
+polynomial_decay_pruning_schedule/Minimum/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+polynomial_decay_pruning_schedule/Minimum/xç
)polynomial_decay_pruning_schedule/MinimumMinimum4polynomial_decay_pruning_schedule/Minimum/x:output:0-polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/Minimum
)polynomial_decay_pruning_schedule/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)polynomial_decay_pruning_schedule/sub_1/xÝ
'polynomial_decay_pruning_schedule/sub_1Sub2polynomial_decay_pruning_schedule/sub_1/x:output:0-polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2)
'polynomial_decay_pruning_schedule/sub_1
'polynomial_decay_pruning_schedule/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2)
'polynomial_decay_pruning_schedule/Pow/yÕ
%polynomial_decay_pruning_schedule/PowPow+polynomial_decay_pruning_schedule/sub_1:z:00polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/Pow
'polynomial_decay_pruning_schedule/Mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¾2)
'polynomial_decay_pruning_schedule/Mul/xÓ
%polynomial_decay_pruning_schedule/MulMul0polynomial_decay_pruning_schedule/Mul/x:output:0)polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/Mul¡
,polynomial_decay_pruning_schedule/sparsity/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2.
,polynomial_decay_pruning_schedule/sparsity/yâ
*polynomial_decay_pruning_schedule/sparsityAdd)polynomial_decay_pruning_schedule/Mul:z:05polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2,
*polynomial_decay_pruning_schedule/sparsity¬
GreaterEqual/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOpb
GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqual¦
LessEqual/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp^
LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqualT
Less/xConst*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
Less/xR
Less/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0	*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd
Sub/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOpP
Sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
SubZ

FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorModT
Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1¬
pruning_ops/Abs/ReadVariableOpReadVariableOp'pruning_ops_abs_readvariableop_resource*"
_output_shapes
:*
dtype02 
pruning_ops/Abs/ReadVariableOp~
pruning_ops/AbsAbs&pruning_ops/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:2
pruning_ops/Absg
pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value
B :2
pruning_ops/Sizew
pruning_ops/CastCastpruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
pruning_ops/Castk
pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
pruning_ops/sub/x
pruning_ops/subSubpruning_ops/sub/x:output:0.polynomial_decay_pruning_schedule/sparsity:z:0*
T0*
_output_shapes
: 2
pruning_ops/subu
pruning_ops/mulMulpruning_ops/Cast:y:0pruning_ops/sub:z:0*
T0*
_output_shapes
: 2
pruning_ops/mule
pruning_ops/RoundRoundpruning_ops/mul:z:0*
T0*
_output_shapes
: 2
pruning_ops/Rounds
pruning_ops/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
pruning_ops/Maximum/y
pruning_ops/MaximumMaximumpruning_ops/Round:y:0pruning_ops/Maximum/y:output:0*
T0*
_output_shapes
: 2
pruning_ops/Maximumy
pruning_ops/Cast_1Castpruning_ops/Maximum:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
pruning_ops/Cast_1
pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
pruning_ops/Reshape/shape
pruning_ops/ReshapeReshapepruning_ops/Abs:y:0"pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes	
:2
pruning_ops/Reshapek
pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value
B :2
pruning_ops/Size_1
pruning_ops/TopKV2TopKV2pruning_ops/Reshape:output:0pruning_ops/Size_1:output:0*
T0*"
_output_shapes
::2
pruning_ops/TopKV2l
pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
pruning_ops/sub_1/y
pruning_ops/sub_1Subpruning_ops/Cast_1:y:0pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: 2
pruning_ops/sub_1x
pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
pruning_ops/GatherV2/axisÔ
pruning_ops/GatherV2GatherV2pruning_ops/TopKV2:values:0pruning_ops/sub_1:z:0"pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: 2
pruning_ops/GatherV2¥
pruning_ops/GreaterEqualGreaterEqualpruning_ops/Abs:y:0pruning_ops/GatherV2:output:0*
T0*"
_output_shapes
:2
pruning_ops/GreaterEqual
pruning_ops/Cast_2Castpruning_ops/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
:2
pruning_ops/Cast_2
AssignVariableOpAssignVariableOpassignvariableop_resourcepruning_ops/Cast_2:y:0*
_output_shapes
 *
dtype02
AssignVariableOp
AssignVariableOp_1AssignVariableOpassignvariableop_1_resourcepruning_ops/GatherV2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1

group_depsNoOp^AssignVariableOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_deps
IdentityIdentity2identity_prune_low_magnitude_conv1d_1_logicaland_1^group_deps*
T0
*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^AssignVariableOp^AssignVariableOp_1*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
::::: 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_1: 

_output_shapes
: :
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
: 
Ó
v
-prune_low_magnitude_dropout_cond_false_3179765
1identity_prune_low_magnitude_dropout_logicaland_1


identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOp{
IdentityIdentity1identity_prune_low_magnitude_dropout_logicaland_1^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : 

_output_shapes
: 
Â5


F__inference_sequential_layer_call_and_return_conditional_losses_317164
conv1d_input%
!prune_low_magnitude_conv1d_316248%
!prune_low_magnitude_conv1d_316250%
!prune_low_magnitude_conv1d_316252%
!prune_low_magnitude_conv1d_316254%
!prune_low_magnitude_conv1d_316256'
#prune_low_magnitude_conv1d_1_316477'
#prune_low_magnitude_conv1d_1_316479'
#prune_low_magnitude_conv1d_1_316481'
#prune_low_magnitude_conv1d_1_316483'
#prune_low_magnitude_conv1d_1_316485'
#prune_low_magnitude_conv1d_2_316706'
#prune_low_magnitude_conv1d_2_316708'
#prune_low_magnitude_conv1d_2_316710'
#prune_low_magnitude_conv1d_2_316712'
#prune_low_magnitude_conv1d_2_316714&
"prune_low_magnitude_dropout_316830&
"prune_low_magnitude_flatten_316941$
 prune_low_magnitude_dense_317152$
 prune_low_magnitude_dense_317154$
 prune_low_magnitude_dense_317156$
 prune_low_magnitude_dense_317158$
 prune_low_magnitude_dense_317160
identity¢2prune_low_magnitude_conv1d/StatefulPartitionedCall¢4prune_low_magnitude_conv1d_1/StatefulPartitionedCall¢4prune_low_magnitude_conv1d_2/StatefulPartitionedCall¢1prune_low_magnitude_dense/StatefulPartitionedCall¢3prune_low_magnitude_dropout/StatefulPartitionedCall¢3prune_low_magnitude_flatten/StatefulPartitionedCallË
2prune_low_magnitude_conv1d/StatefulPartitionedCallStatefulPartitionedCallconv1d_input!prune_low_magnitude_conv1d_316248!prune_low_magnitude_conv1d_316250!prune_low_magnitude_conv1d_316252!prune_low_magnitude_conv1d_316254!prune_low_magnitude_conv1d_316256*
Tin

2*
Tout
2*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	*#
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*_
fZRX
V__inference_prune_low_magnitude_conv1d_layer_call_and_return_conditional_losses_31620024
2prune_low_magnitude_conv1d/StatefulPartitionedCall
4prune_low_magnitude_conv1d_1/StatefulPartitionedCallStatefulPartitionedCall;prune_low_magnitude_conv1d/StatefulPartitionedCall:output:0#prune_low_magnitude_conv1d_1_316477#prune_low_magnitude_conv1d_1_316479#prune_low_magnitude_conv1d_1_316481#prune_low_magnitude_conv1d_1_316483#prune_low_magnitude_conv1d_1_316485*
Tin

2*
Tout
2*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	*#
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*a
f\RZ
X__inference_prune_low_magnitude_conv1d_1_layer_call_and_return_conditional_losses_31642926
4prune_low_magnitude_conv1d_1/StatefulPartitionedCall
4prune_low_magnitude_conv1d_2/StatefulPartitionedCallStatefulPartitionedCall=prune_low_magnitude_conv1d_1/StatefulPartitionedCall:output:0#prune_low_magnitude_conv1d_2_316706#prune_low_magnitude_conv1d_2_316708#prune_low_magnitude_conv1d_2_316710#prune_low_magnitude_conv1d_2_316712#prune_low_magnitude_conv1d_2_316714*
Tin

2*
Tout
2*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	*#
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*a
f\RZ
X__inference_prune_low_magnitude_conv1d_2_layer_call_and_return_conditional_losses_31665826
4prune_low_magnitude_conv1d_2/StatefulPartitionedCallí
3prune_low_magnitude_dropout/StatefulPartitionedCallStatefulPartitionedCall=prune_low_magnitude_conv1d_2/StatefulPartitionedCall:output:0"prune_low_magnitude_dropout_316830*
Tin
2*
Tout
2*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	* 
_read_only_resource_inputs
 *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*`
f[RY
W__inference_prune_low_magnitude_dropout_layer_call_and_return_conditional_losses_31681125
3prune_low_magnitude_dropout/StatefulPartitionedCallé
3prune_low_magnitude_flatten/StatefulPartitionedCallStatefulPartitionedCall<prune_low_magnitude_dropout/StatefulPartitionedCall:output:0"prune_low_magnitude_flatten_316941*
Tin
2*
Tout
2*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*`
f[RY
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_31692125
3prune_low_magnitude_flatten/StatefulPartitionedCallî
1prune_low_magnitude_dense/StatefulPartitionedCallStatefulPartitionedCall<prune_low_magnitude_flatten/StatefulPartitionedCall:output:0 prune_low_magnitude_dense_317152 prune_low_magnitude_dense_317154 prune_low_magnitude_dense_317156 prune_low_magnitude_dense_317158 prune_low_magnitude_dense_317160*
Tin

2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*^
fYRW
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_31710923
1prune_low_magnitude_dense/StatefulPartitionedCallÑ
IdentityIdentity:prune_low_magnitude_dense/StatefulPartitionedCall:output:03^prune_low_magnitude_conv1d/StatefulPartitionedCall5^prune_low_magnitude_conv1d_1/StatefulPartitionedCall5^prune_low_magnitude_conv1d_2/StatefulPartitionedCall2^prune_low_magnitude_dense/StatefulPartitionedCall4^prune_low_magnitude_dropout/StatefulPartitionedCall4^prune_low_magnitude_flatten/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesr
p:ÿÿÿÿÿÿÿÿÿâ	::::::::::::::::::::::2h
2prune_low_magnitude_conv1d/StatefulPartitionedCall2prune_low_magnitude_conv1d/StatefulPartitionedCall2l
4prune_low_magnitude_conv1d_1/StatefulPartitionedCall4prune_low_magnitude_conv1d_1/StatefulPartitionedCall2l
4prune_low_magnitude_conv1d_2/StatefulPartitionedCall4prune_low_magnitude_conv1d_2/StatefulPartitionedCall2f
1prune_low_magnitude_dense/StatefulPartitionedCall1prune_low_magnitude_dense/StatefulPartitionedCall2j
3prune_low_magnitude_dropout/StatefulPartitionedCall3prune_low_magnitude_dropout/StatefulPartitionedCall2j
3prune_low_magnitude_flatten/StatefulPartitionedCall3prune_low_magnitude_flatten/StatefulPartitionedCall:Z V
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	
&
_user_specified_nameconv1d_input:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ñ
ñ
V__inference_prune_low_magnitude_conv1d_layer_call_and_return_conditional_losses_316220

inputs
mul_readvariableop_resource!
mul_readvariableop_1_resource#
biasadd_readvariableop_resource
identity¢AssignVariableOp4
	no_updateNoOp*
_output_shapes
 2
	no_update8
no_update_1NoOp*
_output_shapes
 2
no_update_1
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*"
_output_shapes
:*
dtype02
Mul/ReadVariableOp
Mul/ReadVariableOp_1ReadVariableOpmul_readvariableop_1_resource*"
_output_shapes
:*
dtype02
Mul/ReadVariableOp_1x
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*"
_output_shapes
:2
Mul
AssignVariableOpAssignVariableOpmul_readvariableop_resourceMul:z:0^Mul/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpw

group_depsNoOp^AssignVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_depsu
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2
group_deps_1p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	2
conv1d/ExpandDims»
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1¸
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	*
squeeze_dims
2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	2
Relu~
IdentityIdentityRelu:activations:0^AssignVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿâ	:::2$
AssignVariableOpAssignVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


3assert_greater_equal_Assert_AssertGuard_true_318833%
!identity_assert_greater_equal_all

placeholder	
placeholder_1	

identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOpk
IdentityIdentity!identity_assert_greater_equal_all^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ù
¡
=__inference_prune_low_magnitude_conv1d_2_layer_call_fn_319033

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	*$
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*a
f\RZ
X__inference_prune_low_magnitude_conv1d_2_layer_call_and_return_conditional_losses_3166782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÝ	:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


3assert_greater_equal_Assert_AssertGuard_true_316960%
!identity_assert_greater_equal_all

placeholder	
placeholder_1	

identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOpk
IdentityIdentity!identity_assert_greater_equal_all^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ã
Ê
4assert_greater_equal_Assert_AssertGuard_false_319161#
assert_assert_greater_equal_all
.
*assert_assert_greater_equal_readvariableop	!
assert_assert_greater_equal_y	

identity_1
¢Assertî
Assert/data_0Const*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2
Assert/data_0
Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2
Assert/data_1
Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2
Assert/data_2~
Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2
Assert/data_4
AssertAssertassert_assert_greater_equal_allAssert/data_0:output:0Assert/data_1:output:0Assert/data_2:output:0*assert_assert_greater_equal_readvariableopAssert/data_4:output:0assert_assert_greater_equal_y*
T

2		*
_output_shapes
 2
Assertk
IdentityIdentityassert_assert_greater_equal_all^Assert*
T0
*
_output_shapes
: 2

Identitya

Identity_1IdentityIdentity:output:0^Assert*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : 2
AssertAssert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ó
s
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_316927

inputs
identity4
	no_updateNoOp*
_output_shapes
 2
	no_update8
no_update_1NoOp*
_output_shapes
 2
no_update_16

group_depsNoOp*
_output_shapes
 2

group_deps_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿM  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÙ	:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	
 
_user_specified_nameinputs
Ò

cond_false_318459
placeholder
placeholder_1
placeholder_2
placeholder_3
identity_logicaland_1


identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOp_
IdentityIdentityidentity_logicaland_1^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
::::: : 

_output_shapes
: :
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
: 


3assert_greater_equal_Assert_AssertGuard_true_319160%
!identity_assert_greater_equal_all

placeholder	
placeholder_1	

identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOpk
IdentityIdentity!identity_assert_greater_equal_all^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ûf
È
X__inference_prune_low_magnitude_conv1d_1_layer_call_and_return_conditional_losses_316429

inputs
readvariableop_resource
cond_input_1
cond_input_2
cond_input_3#
biasadd_readvariableop_resource
identity¢AssignVariableOp¢AssignVariableOp_1¢'assert_greater_equal/Assert/AssertGuard¢condp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOpP
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
add/y\
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: 2
add
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOpA
updateNoOp^AssignVariableOp*
_output_shapes
 2
update­
#assert_greater_equal/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
: *
dtype0	2%
#assert_greater_equal/ReadVariableOpr
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
assert_greater_equal/yÅ
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2#
!assert_greater_equal/GreaterEqual{
assert_greater_equal/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
assert_greater_equal/Const
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/Const:output:0*
_output_shapes
: 2
assert_greater_equal/All
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2#
!assert_greater_equal/Assert/Const¶
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2%
#assert_greater_equal/Assert/Const_1·
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2%
#assert_greater_equal/Assert/Const_2ª
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2%
#assert_greater_equal/Assert/Const_3
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_316276*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_3162752)
'assert_greater_equal/Assert/AssertGuardÃ
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identityú
0polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	22
0polynomial_decay_pruning_schedule/ReadVariableOpÇ
'polynomial_decay_pruning_schedule/sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'polynomial_decay_pruning_schedule/sub/yâ
%polynomial_decay_pruning_schedule/subSub8polynomial_decay_pruning_schedule/ReadVariableOp:value:00polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/sub³
&polynomial_decay_pruning_schedule/CastCast)polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2(
&polynomial_decay_pruning_schedule/CastÒ
+polynomial_decay_pruning_schedule/truediv/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 * ¨G2-
+polynomial_decay_pruning_schedule/truediv/yä
)polynomial_decay_pruning_schedule/truedivRealDiv*polynomial_decay_pruning_schedule/Cast:y:04polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/truedivÒ
+polynomial_decay_pruning_schedule/Maximum/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *    2-
+polynomial_decay_pruning_schedule/Maximum/xç
)polynomial_decay_pruning_schedule/MaximumMaximum4polynomial_decay_pruning_schedule/Maximum/x:output:0-polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/MaximumÒ
+polynomial_decay_pruning_schedule/Minimum/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+polynomial_decay_pruning_schedule/Minimum/xç
)polynomial_decay_pruning_schedule/MinimumMinimum4polynomial_decay_pruning_schedule/Minimum/x:output:0-polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/MinimumÎ
)polynomial_decay_pruning_schedule/sub_1/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)polynomial_decay_pruning_schedule/sub_1/xÝ
'polynomial_decay_pruning_schedule/sub_1Sub2polynomial_decay_pruning_schedule/sub_1/x:output:0-polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2)
'polynomial_decay_pruning_schedule/sub_1Ê
'polynomial_decay_pruning_schedule/Pow/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  @@2)
'polynomial_decay_pruning_schedule/Pow/yÕ
%polynomial_decay_pruning_schedule/PowPow+polynomial_decay_pruning_schedule/sub_1:z:00polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/PowÊ
'polynomial_decay_pruning_schedule/Mul/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *¾2)
'polynomial_decay_pruning_schedule/Mul/xÓ
%polynomial_decay_pruning_schedule/MulMul0polynomial_decay_pruning_schedule/Mul/x:output:0)polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/MulÔ
,polynomial_decay_pruning_schedule/sparsity/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2.
,polynomial_decay_pruning_schedule/sparsity/yâ
*polynomial_decay_pruning_schedule/sparsityAdd)polynomial_decay_pruning_schedule/Mul:z:05polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2,
*polynomial_decay_pruning_schedule/sparsityÐ
GreaterEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOp
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqualÊ
LessEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqual
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
Less/x
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0	*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd¾
Sub/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOp
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
Sub

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorMod
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1û
condIfLogicalAnd_1:z:0readvariableop_resourcecond_input_1cond_input_2cond_input_3LogicalAnd_1:z:0^AssignVariableOp*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*$
else_branchR
cond_false_316333*
output_shapes
: *#
then_branchR
cond_true_3163322
condZ
cond/IdentityIdentitycond:output:0*
T0
*
_output_shapes
: 2
cond/Identityu
update_1NoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 2

update_1y
Mul/ReadVariableOpReadVariableOpcond_input_1*"
_output_shapes
:*
dtype02
Mul/ReadVariableOp
Mul/ReadVariableOp_1ReadVariableOpcond_input_2^cond*"
_output_shapes
:*
dtype02
Mul/ReadVariableOp_1x
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*"
_output_shapes
:2
Mul
AssignVariableOp_1AssignVariableOpcond_input_1Mul:z:0^Mul/ReadVariableOp^cond*
_output_shapes
 *
dtype02
AssignVariableOp_1y

group_depsNoOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_depsu
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2
group_deps_1p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	2
conv1d/ExpandDims®
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOpcond_input_1^AssignVariableOp_1*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1¸
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	*
squeeze_dims
2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	2
ReluÄ
IdentityIdentityRelu:activations:0^AssignVariableOp^AssignVariableOp_1(^assert_greater_equal/Assert/AssertGuard^cond*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿà	:::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2
condcond:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	
 
_user_specified_nameinputs:
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
: 
ó
¶
Oprune_low_magnitude_conv1d_assert_greater_equal_Assert_AssertGuard_false_317421>
:assert_prune_low_magnitude_conv1d_assert_greater_equal_all
I
Eassert_prune_low_magnitude_conv1d_assert_greater_equal_readvariableop	<
8assert_prune_low_magnitude_conv1d_assert_greater_equal_y	

identity_1
¢Assertî
Assert/data_0Const*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2
Assert/data_0
Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2
Assert/data_1¦
Assert/data_2Const*
_output_shapes
: *
dtype0*X
valueOBM BGx (prune_low_magnitude_conv1d/assert_greater_equal/ReadVariableOp:0) = 2
Assert/data_2
Assert/data_4Const*
_output_shapes
: *
dtype0*K
valueBB@ B:y (prune_low_magnitude_conv1d/assert_greater_equal/y:0) = 2
Assert/data_4Þ
AssertAssert:assert_prune_low_magnitude_conv1d_assert_greater_equal_allAssert/data_0:output:0Assert/data_1:output:0Assert/data_2:output:0Eassert_prune_low_magnitude_conv1d_assert_greater_equal_readvariableopAssert/data_4:output:08assert_prune_low_magnitude_conv1d_assert_greater_equal_y*
T

2		*
_output_shapes
 2
Assert
IdentityIdentity:assert_prune_low_magnitude_conv1d_assert_greater_equal_all^Assert*
T0
*
_output_shapes
: 2

Identitya

Identity_1IdentityIdentity:output:0^Assert*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : 2
AssertAssert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ûS
Ð
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_316921

inputs
readvariableop_resource
identity¢AssignVariableOp¢'assert_greater_equal/Assert/AssertGuardp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOpP
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
add/y\
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: 2
add
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOpA
updateNoOp^AssignVariableOp*
_output_shapes
 2
update­
#assert_greater_equal/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
: *
dtype0	2%
#assert_greater_equal/ReadVariableOpr
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
assert_greater_equal/yÅ
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2#
!assert_greater_equal/GreaterEqual{
assert_greater_equal/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
assert_greater_equal/Const
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/Const:output:0*
_output_shapes
: 2
assert_greater_equal/All
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2#
!assert_greater_equal/Assert/Const¶
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2%
#assert_greater_equal/Assert/Const_1·
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2%
#assert_greater_equal/Assert/Const_2ª
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2%
#assert_greater_equal/Assert/Const_3
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_316850*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_3168492)
'assert_greater_equal/Assert/AssertGuardÃ
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identityú
0polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	22
0polynomial_decay_pruning_schedule/ReadVariableOpÇ
'polynomial_decay_pruning_schedule/sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'polynomial_decay_pruning_schedule/sub/yâ
%polynomial_decay_pruning_schedule/subSub8polynomial_decay_pruning_schedule/ReadVariableOp:value:00polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/sub³
&polynomial_decay_pruning_schedule/CastCast)polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2(
&polynomial_decay_pruning_schedule/CastÒ
+polynomial_decay_pruning_schedule/truediv/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 * ¨G2-
+polynomial_decay_pruning_schedule/truediv/yä
)polynomial_decay_pruning_schedule/truedivRealDiv*polynomial_decay_pruning_schedule/Cast:y:04polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/truedivÒ
+polynomial_decay_pruning_schedule/Maximum/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *    2-
+polynomial_decay_pruning_schedule/Maximum/xç
)polynomial_decay_pruning_schedule/MaximumMaximum4polynomial_decay_pruning_schedule/Maximum/x:output:0-polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/MaximumÒ
+polynomial_decay_pruning_schedule/Minimum/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+polynomial_decay_pruning_schedule/Minimum/xç
)polynomial_decay_pruning_schedule/MinimumMinimum4polynomial_decay_pruning_schedule/Minimum/x:output:0-polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/MinimumÎ
)polynomial_decay_pruning_schedule/sub_1/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)polynomial_decay_pruning_schedule/sub_1/xÝ
'polynomial_decay_pruning_schedule/sub_1Sub2polynomial_decay_pruning_schedule/sub_1/x:output:0-polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2)
'polynomial_decay_pruning_schedule/sub_1Ê
'polynomial_decay_pruning_schedule/Pow/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  @@2)
'polynomial_decay_pruning_schedule/Pow/yÕ
%polynomial_decay_pruning_schedule/PowPow+polynomial_decay_pruning_schedule/sub_1:z:00polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/PowÊ
'polynomial_decay_pruning_schedule/Mul/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *¾2)
'polynomial_decay_pruning_schedule/Mul/xÓ
%polynomial_decay_pruning_schedule/MulMul0polynomial_decay_pruning_schedule/Mul/x:output:0)polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/MulÔ
,polynomial_decay_pruning_schedule/sparsity/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2.
,polynomial_decay_pruning_schedule/sparsity/yâ
*polynomial_decay_pruning_schedule/sparsityAdd)polynomial_decay_pruning_schedule/Mul:z:05polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2,
*polynomial_decay_pruning_schedule/sparsityÐ
GreaterEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOp
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqualÊ
LessEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqual
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
Less/x
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0	*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd¾
Sub/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOp
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
Sub

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorMod
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1¦
condStatelessIfLogicalAnd_1:z:0LogicalAnd_1:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *$
else_branchR
cond_false_316907*
output_shapes
: *#
then_branchR
cond_true_3169062
condZ
cond/IdentityIdentitycond:output:0*
T0
*
_output_shapes
: 2
cond/Identityu
update_1NoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 2

update_16

group_depsNoOp*
_output_shapes
 2

group_deps_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿM  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshape£
IdentityIdentityReshape:output:0^AssignVariableOp(^assert_greater_equal/Assert/AssertGuard*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÙ	:2$
AssignVariableOpAssignVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	
 
_user_specified_nameinputs:

_output_shapes
: 
ë
·
+__inference_sequential_layer_call_fn_318356

inputs
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
identity¢StatefulPartitionedCallÚ
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
unknown_20*"
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs

*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3172552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesr
p:ÿÿÿÿÿÿÿÿÿâ	::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	
 
_user_specified_nameinputs:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

=
cond_true_319217
identity_logicaland_1


identity_1
6

group_depsNoOp*
_output_shapes
 2

group_depse
IdentityIdentityidentity_logicaland_1^group_deps*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : 

_output_shapes
: 

=
cond_true_316906
identity_logicaland_1


identity_1
6

group_depsNoOp*
_output_shapes
 2

group_depse
IdentityIdentityidentity_logicaland_1^group_deps*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : 

_output_shapes
: 
ó
ó
X__inference_prune_low_magnitude_conv1d_1_layer_call_and_return_conditional_losses_316449

inputs
mul_readvariableop_resource!
mul_readvariableop_1_resource#
biasadd_readvariableop_resource
identity¢AssignVariableOp4
	no_updateNoOp*
_output_shapes
 2
	no_update8
no_update_1NoOp*
_output_shapes
 2
no_update_1
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*"
_output_shapes
:*
dtype02
Mul/ReadVariableOp
Mul/ReadVariableOp_1ReadVariableOpmul_readvariableop_1_resource*"
_output_shapes
:*
dtype02
Mul/ReadVariableOp_1x
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*"
_output_shapes
:2
Mul
AssignVariableOpAssignVariableOpmul_readvariableop_resourceMul:z:0^Mul/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpw

group_depsNoOp^AssignVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_depsu
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2
group_deps_1p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	2
conv1d/ExpandDims»
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1¸
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	*
squeeze_dims
2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	2
Relu~
IdentityIdentityRelu:activations:0^AssignVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿà	:::2$
AssignVariableOpAssignVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

·
B__inference_conv1d_layer_call_and_return_conditional_losses_315962

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityp
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1À
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ñ(

F__inference_sequential_layer_call_and_return_conditional_losses_317338

inputs%
!prune_low_magnitude_conv1d_317307%
!prune_low_magnitude_conv1d_317309%
!prune_low_magnitude_conv1d_317311'
#prune_low_magnitude_conv1d_1_317314'
#prune_low_magnitude_conv1d_1_317316'
#prune_low_magnitude_conv1d_1_317318'
#prune_low_magnitude_conv1d_2_317321'
#prune_low_magnitude_conv1d_2_317323'
#prune_low_magnitude_conv1d_2_317325$
 prune_low_magnitude_dense_317330$
 prune_low_magnitude_dense_317332$
 prune_low_magnitude_dense_317334
identity¢2prune_low_magnitude_conv1d/StatefulPartitionedCall¢4prune_low_magnitude_conv1d_1/StatefulPartitionedCall¢4prune_low_magnitude_conv1d_2/StatefulPartitionedCall¢1prune_low_magnitude_dense/StatefulPartitionedCallþ
2prune_low_magnitude_conv1d/StatefulPartitionedCallStatefulPartitionedCallinputs!prune_low_magnitude_conv1d_317307!prune_low_magnitude_conv1d_317309!prune_low_magnitude_conv1d_317311*
Tin
2*
Tout
2*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà	*$
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*_
fZRX
V__inference_prune_low_magnitude_conv1d_layer_call_and_return_conditional_losses_31622024
2prune_low_magnitude_conv1d/StatefulPartitionedCall¿
4prune_low_magnitude_conv1d_1/StatefulPartitionedCallStatefulPartitionedCall;prune_low_magnitude_conv1d/StatefulPartitionedCall:output:0#prune_low_magnitude_conv1d_1_317314#prune_low_magnitude_conv1d_1_317316#prune_low_magnitude_conv1d_1_317318*
Tin
2*
Tout
2*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	*$
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*a
f\RZ
X__inference_prune_low_magnitude_conv1d_1_layer_call_and_return_conditional_losses_31644926
4prune_low_magnitude_conv1d_1/StatefulPartitionedCallÁ
4prune_low_magnitude_conv1d_2/StatefulPartitionedCallStatefulPartitionedCall=prune_low_magnitude_conv1d_1/StatefulPartitionedCall:output:0#prune_low_magnitude_conv1d_2_317321#prune_low_magnitude_conv1d_2_317323#prune_low_magnitude_conv1d_2_317325*
Tin
2*
Tout
2*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	*$
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*a
f\RZ
X__inference_prune_low_magnitude_conv1d_2_layer_call_and_return_conditional_losses_31667826
4prune_low_magnitude_conv1d_2/StatefulPartitionedCall°
+prune_low_magnitude_dropout/PartitionedCallPartitionedCall=prune_low_magnitude_conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	* 
_read_only_resource_inputs
 *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*`
f[RY
W__inference_prune_low_magnitude_dropout_layer_call_and_return_conditional_losses_3168162-
+prune_low_magnitude_dropout/PartitionedCall¤
+prune_low_magnitude_flatten/PartitionedCallPartitionedCall4prune_low_magnitude_dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*`
f[RY
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_3169272-
+prune_low_magnitude_flatten/PartitionedCall¡
1prune_low_magnitude_dense/StatefulPartitionedCallStatefulPartitionedCall4prune_low_magnitude_flatten/PartitionedCall:output:0 prune_low_magnitude_dense_317330 prune_low_magnitude_dense_317332 prune_low_magnitude_dense_317334*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*^
fYRW
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_31712423
1prune_low_magnitude_dense/StatefulPartitionedCallå
IdentityIdentity:prune_low_magnitude_dense/StatefulPartitionedCall:output:03^prune_low_magnitude_conv1d/StatefulPartitionedCall5^prune_low_magnitude_conv1d_1/StatefulPartitionedCall5^prune_low_magnitude_conv1d_2/StatefulPartitionedCall2^prune_low_magnitude_dense/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿâ	::::::::::::2h
2prune_low_magnitude_conv1d/StatefulPartitionedCall2prune_low_magnitude_conv1d/StatefulPartitionedCall2l
4prune_low_magnitude_conv1d_1/StatefulPartitionedCall4prune_low_magnitude_conv1d_1/StatefulPartitionedCall2l
4prune_low_magnitude_conv1d_2/StatefulPartitionedCall4prune_low_magnitude_conv1d_2/StatefulPartitionedCall2f
1prune_low_magnitude_dense/StatefulPartitionedCall1prune_low_magnitude_dense/StatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	
 
_user_specified_nameinputs:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


3assert_greater_equal_Assert_AssertGuard_true_316849%
!identity_assert_greater_equal_all

placeholder	
placeholder_1	

identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOpk
IdentityIdentity!identity_assert_greater_equal_all^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


3assert_greater_equal_Assert_AssertGuard_true_318617%
!identity_assert_greater_equal_all

placeholder	
placeholder_1	

identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOpk
IdentityIdentity!identity_assert_greater_equal_all^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ó
v
-prune_low_magnitude_flatten_cond_false_3180665
1identity_prune_low_magnitude_flatten_logicaland_1


identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOp{
IdentityIdentity1identity_prune_low_magnitude_flatten_logicaland_1^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : 

_output_shapes
: 
ã
Ê
4assert_greater_equal_Assert_AssertGuard_false_316850#
assert_assert_greater_equal_all
.
*assert_assert_greater_equal_readvariableop	!
assert_assert_greater_equal_y	

identity_1
¢Assertî
Assert/data_0Const*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2
Assert/data_0
Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2
Assert/data_1
Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2
Assert/data_2~
Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2
Assert/data_4
AssertAssertassert_assert_greater_equal_allAssert/data_0:output:0Assert/data_1:output:0Assert/data_2:output:0*assert_assert_greater_equal_readvariableopAssert/data_4:output:0assert_assert_greater_equal_y*
T

2		*
_output_shapes
 2
Assertk
IdentityIdentityassert_assert_greater_equal_all^Assert*
T0
*
_output_shapes
: 2

Identitya

Identity_1IdentityIdentity:output:0^Assert*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : 2
AssertAssert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ó
ó
X__inference_prune_low_magnitude_conv1d_2_layer_call_and_return_conditional_losses_319007

inputs
mul_readvariableop_resource!
mul_readvariableop_1_resource#
biasadd_readvariableop_resource
identity¢AssignVariableOp4
	no_updateNoOp*
_output_shapes
 2
	no_update8
no_update_1NoOp*
_output_shapes
 2
no_update_1
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*"
_output_shapes
:*
dtype02
Mul/ReadVariableOp
Mul/ReadVariableOp_1ReadVariableOpmul_readvariableop_1_resource*"
_output_shapes
:*
dtype02
Mul/ReadVariableOp_1x
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*"
_output_shapes
:2
Mul
AssignVariableOpAssignVariableOpmul_readvariableop_resourceMul:z:0^Mul/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpw

group_depsNoOp^AssignVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_depsu
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2
group_deps_1p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	2
conv1d/ExpandDims»
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1¸
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	*
squeeze_dims
2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2
Relu~
IdentityIdentityRelu:activations:0^AssignVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÝ	:::2$
AssignVariableOpAssignVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ë

"__inference__traced_restore_319777
file_prefix4
0assignvariableop_prune_low_magnitude_conv1d_mask;
7assignvariableop_1_prune_low_magnitude_conv1d_threshold>
:assignvariableop_2_prune_low_magnitude_conv1d_pruning_step8
4assignvariableop_3_prune_low_magnitude_conv1d_1_mask=
9assignvariableop_4_prune_low_magnitude_conv1d_1_threshold@
<assignvariableop_5_prune_low_magnitude_conv1d_1_pruning_step8
4assignvariableop_6_prune_low_magnitude_conv1d_2_mask=
9assignvariableop_7_prune_low_magnitude_conv1d_2_threshold@
<assignvariableop_8_prune_low_magnitude_conv1d_2_pruning_step?
;assignvariableop_9_prune_low_magnitude_dropout_pruning_step@
<assignvariableop_10_prune_low_magnitude_flatten_pruning_step6
2assignvariableop_11_prune_low_magnitude_dense_mask;
7assignvariableop_12_prune_low_magnitude_dense_threshold>
:assignvariableop_13_prune_low_magnitude_dense_pruning_step
assignvariableop_14_iter#
assignvariableop_15_adam_beta_1#
assignvariableop_16_adam_beta_2"
assignvariableop_17_adam_decay*
&assignvariableop_18_adam_learning_rate%
!assignvariableop_19_conv1d_kernel#
assignvariableop_20_conv1d_bias'
#assignvariableop_21_conv1d_1_kernel%
!assignvariableop_22_conv1d_1_bias'
#assignvariableop_23_conv1d_2_kernel%
!assignvariableop_24_conv1d_2_bias$
 assignvariableop_25_dense_kernel"
assignvariableop_26_dense_bias
assignvariableop_27_total
assignvariableop_28_count
assignvariableop_29_total_1
assignvariableop_30_count_1,
(assignvariableop_31_adam_conv1d_kernel_m*
&assignvariableop_32_adam_conv1d_bias_m.
*assignvariableop_33_adam_conv1d_1_kernel_m,
(assignvariableop_34_adam_conv1d_1_bias_m.
*assignvariableop_35_adam_conv1d_2_kernel_m,
(assignvariableop_36_adam_conv1d_2_bias_m+
'assignvariableop_37_adam_dense_kernel_m)
%assignvariableop_38_adam_dense_bias_m,
(assignvariableop_39_adam_conv1d_kernel_v*
&assignvariableop_40_adam_conv1d_bias_v.
*assignvariableop_41_adam_conv1d_1_kernel_v,
(assignvariableop_42_adam_conv1d_1_bias_v.
*assignvariableop_43_adam_conv1d_2_kernel_v,
(assignvariableop_44_adam_conv1d_2_bias_v+
'assignvariableop_45_adam_dense_kernel_v)
%assignvariableop_46_adam_dense_bias_v
identity_48¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¢	RestoreV2¢RestoreV2_1î
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*ú
valueðBí/B4layer_with_weights-0/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-0/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-1/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-1/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-2/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-2/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-3/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-4/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-5/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-5/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesì
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ò
_output_shapes¿
¼:::::::::::::::::::::::::::::::::::::::::::::::*=
dtypes3
12/							2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity 
AssignVariableOpAssignVariableOp0assignvariableop_prune_low_magnitude_conv1d_maskIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1­
AssignVariableOp_1AssignVariableOp7assignvariableop_1_prune_low_magnitude_conv1d_thresholdIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0	*
_output_shapes
:2

Identity_2°
AssignVariableOp_2AssignVariableOp:assignvariableop_2_prune_low_magnitude_conv1d_pruning_stepIdentity_2:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3ª
AssignVariableOp_3AssignVariableOp4assignvariableop_3_prune_low_magnitude_conv1d_1_maskIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4¯
AssignVariableOp_4AssignVariableOp9assignvariableop_4_prune_low_magnitude_conv1d_1_thresholdIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0	*
_output_shapes
:2

Identity_5²
AssignVariableOp_5AssignVariableOp<assignvariableop_5_prune_low_magnitude_conv1d_1_pruning_stepIdentity_5:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6ª
AssignVariableOp_6AssignVariableOp4assignvariableop_6_prune_low_magnitude_conv1d_2_maskIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7¯
AssignVariableOp_7AssignVariableOp9assignvariableop_7_prune_low_magnitude_conv1d_2_thresholdIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0	*
_output_shapes
:2

Identity_8²
AssignVariableOp_8AssignVariableOp<assignvariableop_8_prune_low_magnitude_conv1d_2_pruning_stepIdentity_8:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0	*
_output_shapes
:2

Identity_9±
AssignVariableOp_9AssignVariableOp;assignvariableop_9_prune_low_magnitude_dropout_pruning_stepIdentity_9:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0	*
_output_shapes
:2
Identity_10µ
AssignVariableOp_10AssignVariableOp<assignvariableop_10_prune_low_magnitude_flatten_pruning_stepIdentity_10:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11«
AssignVariableOp_11AssignVariableOp2assignvariableop_11_prune_low_magnitude_dense_maskIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12°
AssignVariableOp_12AssignVariableOp7assignvariableop_12_prune_low_magnitude_dense_thresholdIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0	*
_output_shapes
:2
Identity_13³
AssignVariableOp_13AssignVariableOp:assignvariableop_13_prune_low_magnitude_dense_pruning_stepIdentity_13:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0	*
_output_shapes
:2
Identity_14
AssignVariableOp_14AssignVariableOpassignvariableop_14_iterIdentity_14:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv1d_kernelIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20
AssignVariableOp_20AssignVariableOpassignvariableop_20_conv1d_biasIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21
AssignVariableOp_21AssignVariableOp#assignvariableop_21_conv1d_1_kernelIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22
AssignVariableOp_22AssignVariableOp!assignvariableop_22_conv1d_1_biasIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23
AssignVariableOp_23AssignVariableOp#assignvariableop_23_conv1d_2_kernelIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24
AssignVariableOp_24AssignVariableOp!assignvariableop_24_conv1d_2_biasIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25
AssignVariableOp_25AssignVariableOp assignvariableop_25_dense_kernelIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26
AssignVariableOp_26AssignVariableOpassignvariableop_26_dense_biasIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_1Identity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_1Identity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31¡
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_conv1d_kernel_mIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_conv1d_bias_mIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33£
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv1d_1_kernel_mIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34¡
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv1d_1_bias_mIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35£
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv1d_2_kernel_mIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36¡
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_conv1d_2_bias_mIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37 
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_dense_kernel_mIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38
AssignVariableOp_38AssignVariableOp%assignvariableop_38_adam_dense_bias_mIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39¡
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_conv1d_kernel_vIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40
AssignVariableOp_40AssignVariableOp&assignvariableop_40_adam_conv1d_bias_vIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41£
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv1d_1_kernel_vIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42¡
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv1d_1_bias_vIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43£
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_conv1d_2_kernel_vIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44¡
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv1d_2_bias_vIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45 
AssignVariableOp_45AssignVariableOp'assignvariableop_45_adam_dense_kernel_vIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46
AssignVariableOp_46AssignVariableOp%assignvariableop_46_adam_dense_bias_vIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46¨
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesÄ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpè
Identity_47Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_47õ
Identity_48IdentityIdentity_47:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_48"#
identity_48Identity_48:output:0*Ó
_input_shapesÁ
¾: :::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_46AssignVariableOp_462(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: 
Ò

cond_false_316562
placeholder
placeholder_1
placeholder_2
placeholder_3
identity_logicaland_1


identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOp_
IdentityIdentityidentity_logicaland_1^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
::::: : 

_output_shapes
: :
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
: 
£
¾
,prune_low_magnitude_conv1d_cond_false_317478
placeholder
placeholder_1
placeholder_2
placeholder_34
0identity_prune_low_magnitude_conv1d_logicaland_1


identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOpz
IdentityIdentity0identity_prune_low_magnitude_conv1d_logicaland_1^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
::::: : 

_output_shapes
: :
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
: 
öK
Î
-prune_low_magnitude_conv1d_2_cond_true_317809=
9polynomial_decay_pruning_schedule_readvariableop_resource+
'pruning_ops_abs_readvariableop_resource
assignvariableop_resource
assignvariableop_1_resource6
2identity_prune_low_magnitude_conv1d_2_logicaland_1


identity_1
¢AssignVariableOp¢AssignVariableOp_1Ö
0polynomial_decay_pruning_schedule/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	22
0polynomial_decay_pruning_schedule/ReadVariableOp
'polynomial_decay_pruning_schedule/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'polynomial_decay_pruning_schedule/sub/yâ
%polynomial_decay_pruning_schedule/subSub8polynomial_decay_pruning_schedule/ReadVariableOp:value:00polynomial_decay_pruning_schedule/sub/y:output:0*
T0	*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/sub³
&polynomial_decay_pruning_schedule/CastCast)polynomial_decay_pruning_schedule/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2(
&polynomial_decay_pruning_schedule/Cast
+polynomial_decay_pruning_schedule/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ¨G2-
+polynomial_decay_pruning_schedule/truediv/yä
)polynomial_decay_pruning_schedule/truedivRealDiv*polynomial_decay_pruning_schedule/Cast:y:04polynomial_decay_pruning_schedule/truediv/y:output:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/truediv
+polynomial_decay_pruning_schedule/Maximum/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+polynomial_decay_pruning_schedule/Maximum/xç
)polynomial_decay_pruning_schedule/MaximumMaximum4polynomial_decay_pruning_schedule/Maximum/x:output:0-polynomial_decay_pruning_schedule/truediv:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/Maximum
+polynomial_decay_pruning_schedule/Minimum/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+polynomial_decay_pruning_schedule/Minimum/xç
)polynomial_decay_pruning_schedule/MinimumMinimum4polynomial_decay_pruning_schedule/Minimum/x:output:0-polynomial_decay_pruning_schedule/Maximum:z:0*
T0*
_output_shapes
: 2+
)polynomial_decay_pruning_schedule/Minimum
)polynomial_decay_pruning_schedule/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)polynomial_decay_pruning_schedule/sub_1/xÝ
'polynomial_decay_pruning_schedule/sub_1Sub2polynomial_decay_pruning_schedule/sub_1/x:output:0-polynomial_decay_pruning_schedule/Minimum:z:0*
T0*
_output_shapes
: 2)
'polynomial_decay_pruning_schedule/sub_1
'polynomial_decay_pruning_schedule/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2)
'polynomial_decay_pruning_schedule/Pow/yÕ
%polynomial_decay_pruning_schedule/PowPow+polynomial_decay_pruning_schedule/sub_1:z:00polynomial_decay_pruning_schedule/Pow/y:output:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/Pow
'polynomial_decay_pruning_schedule/Mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¾2)
'polynomial_decay_pruning_schedule/Mul/xÓ
%polynomial_decay_pruning_schedule/MulMul0polynomial_decay_pruning_schedule/Mul/x:output:0)polynomial_decay_pruning_schedule/Pow:z:0*
T0*
_output_shapes
: 2'
%polynomial_decay_pruning_schedule/Mul¡
,polynomial_decay_pruning_schedule/sparsity/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2.
,polynomial_decay_pruning_schedule/sparsity/yâ
*polynomial_decay_pruning_schedule/sparsityAdd)polynomial_decay_pruning_schedule/Mul:z:05polynomial_decay_pruning_schedule/sparsity/y:output:0*
T0*
_output_shapes
: 2,
*polynomial_decay_pruning_schedule/sparsity¬
GreaterEqual/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOpb
GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqual¦
LessEqual/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp^
LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqualT
Less/xConst*
_output_shapes
: *
dtype0	*
valueB		 R¨§2
Less/xR
Less/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0	*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd
Sub/ReadVariableOpReadVariableOp9polynomial_decay_pruning_schedule_readvariableop_resource*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOpP
Sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
SubZ

FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorModT
Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1¬
pruning_ops/Abs/ReadVariableOpReadVariableOp'pruning_ops_abs_readvariableop_resource*"
_output_shapes
:*
dtype02 
pruning_ops/Abs/ReadVariableOp~
pruning_ops/AbsAbs&pruning_ops/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:2
pruning_ops/Absg
pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value
B :
2
pruning_ops/Sizew
pruning_ops/CastCastpruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
pruning_ops/Castk
pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
pruning_ops/sub/x
pruning_ops/subSubpruning_ops/sub/x:output:0.polynomial_decay_pruning_schedule/sparsity:z:0*
T0*
_output_shapes
: 2
pruning_ops/subu
pruning_ops/mulMulpruning_ops/Cast:y:0pruning_ops/sub:z:0*
T0*
_output_shapes
: 2
pruning_ops/mule
pruning_ops/RoundRoundpruning_ops/mul:z:0*
T0*
_output_shapes
: 2
pruning_ops/Rounds
pruning_ops/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
pruning_ops/Maximum/y
pruning_ops/MaximumMaximumpruning_ops/Round:y:0pruning_ops/Maximum/y:output:0*
T0*
_output_shapes
: 2
pruning_ops/Maximumy
pruning_ops/Cast_1Castpruning_ops/Maximum:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
pruning_ops/Cast_1
pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
pruning_ops/Reshape/shape
pruning_ops/ReshapeReshapepruning_ops/Abs:y:0"pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes	
:
2
pruning_ops/Reshapek
pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value
B :
2
pruning_ops/Size_1
pruning_ops/TopKV2TopKV2pruning_ops/Reshape:output:0pruning_ops/Size_1:output:0*
T0*"
_output_shapes
:
:
2
pruning_ops/TopKV2l
pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
pruning_ops/sub_1/y
pruning_ops/sub_1Subpruning_ops/Cast_1:y:0pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: 2
pruning_ops/sub_1x
pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
pruning_ops/GatherV2/axisÔ
pruning_ops/GatherV2GatherV2pruning_ops/TopKV2:values:0pruning_ops/sub_1:z:0"pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: 2
pruning_ops/GatherV2¥
pruning_ops/GreaterEqualGreaterEqualpruning_ops/Abs:y:0pruning_ops/GatherV2:output:0*
T0*"
_output_shapes
:2
pruning_ops/GreaterEqual
pruning_ops/Cast_2Castpruning_ops/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
:2
pruning_ops/Cast_2
AssignVariableOpAssignVariableOpassignvariableop_resourcepruning_ops/Cast_2:y:0*
_output_shapes
 *
dtype02
AssignVariableOp
AssignVariableOp_1AssignVariableOpassignvariableop_1_resourcepruning_ops/GatherV2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1

group_depsNoOp^AssignVariableOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 2

group_deps
IdentityIdentity2identity_prune_low_magnitude_conv1d_2_logicaland_1^group_deps*
T0
*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^AssignVariableOp^AssignVariableOp_1*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
::::: 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_1: 

_output_shapes
: :
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
: 

¹
D__inference_conv1d_2_layer_call_and_return_conditional_losses_316016

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityp
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1À
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
­
X
<__inference_prune_low_magnitude_flatten_layer_call_fn_319250

inputs
identity¾
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*`
f[RY
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_3169272
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÙ	:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ	
 
_user_specified_nameinputs
ÿ
>
cond_false_319107
identity_logicaland_1


identity_1
*
NoOpNoOp*
_output_shapes
 2
NoOp_
IdentityIdentityidentity_logicaland_1^NoOp*
T0
*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : 

_output_shapes
: 
ã
Ê
4assert_greater_equal_Assert_AssertGuard_false_316961#
assert_assert_greater_equal_all
.
*assert_assert_greater_equal_readvariableop	!
assert_assert_greater_equal_y	

identity_1
¢Assertî
Assert/data_0Const*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2
Assert/data_0
Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2
Assert/data_1
Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2
Assert/data_2~
Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2
Assert/data_4
AssertAssertassert_assert_greater_equal_allAssert/data_0:output:0Assert/data_1:output:0Assert/data_2:output:0*assert_assert_greater_equal_readvariableopAssert/data_4:output:0assert_assert_greater_equal_y*
T

2		*
_output_shapes
 2
Assertk
IdentityIdentityassert_assert_greater_equal_all^Assert*
T0
*
_output_shapes
: 2

Identitya

Identity_1IdentityIdentity:output:0^Assert*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : 2
AssertAssert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

¾
Qprune_low_magnitude_conv1d_1_assert_greater_equal_Assert_AssertGuard_false_317587@
<assert_prune_low_magnitude_conv1d_1_assert_greater_equal_all
K
Gassert_prune_low_magnitude_conv1d_1_assert_greater_equal_readvariableop	>
:assert_prune_low_magnitude_conv1d_1_assert_greater_equal_y	

identity_1
¢Assertî
Assert/data_0Const*
_output_shapes
: *
dtype0*
valueB BPrune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2
Assert/data_0
Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2
Assert/data_1¨
Assert/data_2Const*
_output_shapes
: *
dtype0*Z
valueQBO BIx (prune_low_magnitude_conv1d_1/assert_greater_equal/ReadVariableOp:0) = 2
Assert/data_2
Assert/data_4Const*
_output_shapes
: *
dtype0*M
valueDBB B<y (prune_low_magnitude_conv1d_1/assert_greater_equal/y:0) = 2
Assert/data_4ä
AssertAssert<assert_prune_low_magnitude_conv1d_1_assert_greater_equal_allAssert/data_0:output:0Assert/data_1:output:0Assert/data_2:output:0Gassert_prune_low_magnitude_conv1d_1_assert_greater_equal_readvariableopAssert/data_4:output:0:assert_prune_low_magnitude_conv1d_1_assert_greater_equal_y*
T

2		*
_output_shapes
 2
Assert
IdentityIdentity<assert_prune_low_magnitude_conv1d_1_assert_greater_equal_all^Assert*
T0
*
_output_shapes
: 2

Identitya

Identity_1IdentityIdentity:output:0^Assert*
T0
*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : 2
AssertAssert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "¯L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ë
serving_default·
J
conv1d_input:
serving_default_conv1d_input:0ÿÿÿÿÿÿÿÿÿâ	M
prune_low_magnitude_dense0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ì¨
îb
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
	optimizer
regularization_losses
	trainable_variables

	variables
	keras_api

signatures
+Ê&call_and_return_all_conditional_losses
Ë__call__
Ì_default_save_signature"¶_
_tf_keras_sequential_{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "PruneLowMagnitude", "config": {"name": "prune_low_magnitude_conv1d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "dtype": "float32", "layer": {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "pruning_schedule": {"class_name": "PolynomialDecay", "config": {"initial_sparsity": 0.5, "final_sparsity": 0.8, "power": 3, "begin_step": 0, "end_step": 37800, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}}, {"class_name": "PruneLowMagnitude", "config": {"name": "prune_low_magnitude_conv1d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "dtype": "float32", "layer": {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "pruning_schedule": {"class_name": "PolynomialDecay", "config": {"initial_sparsity": 0.5, "final_sparsity": 0.8, "power": 3, "begin_step": 0, "end_step": 37800, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}}, {"class_name": "PruneLowMagnitude", "config": {"name": "prune_low_magnitude_conv1d_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "dtype": "float32", "layer": {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "pruning_schedule": {"class_name": "PolynomialDecay", "config": {"initial_sparsity": 0.5, "final_sparsity": 0.8, "power": 3, "begin_step": 0, "end_step": 37800, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}}, {"class_name": "PruneLowMagnitude", "config": {"name": "prune_low_magnitude_dropout", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, "pruning_schedule": {"class_name": "PolynomialDecay", "config": {"initial_sparsity": 0.5, "final_sparsity": 0.8, "power": 3, "begin_step": 0, "end_step": 37800, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}}, {"class_name": "PruneLowMagnitude", "config": {"name": "prune_low_magnitude_flatten", "trainable": true, "dtype": "float32", "layer": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, "pruning_schedule": {"class_name": "PolynomialDecay", "config": {"initial_sparsity": 0.5, "final_sparsity": 0.8, "power": 3, "begin_step": 0, "end_step": 37800, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}}, {"class_name": "PruneLowMagnitude", "config": {"name": "prune_low_magnitude_dense", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "pruning_schedule": {"class_name": "PolynomialDecay", "config": {"initial_sparsity": 0.5, "final_sparsity": 0.8, "power": 3, "begin_step": 0, "end_step": 37800, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1250, 1]}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1250, 1]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "PruneLowMagnitude", "config": {"name": "prune_low_magnitude_conv1d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "dtype": "float32", "layer": {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "pruning_schedule": {"class_name": "PolynomialDecay", "config": {"initial_sparsity": 0.5, "final_sparsity": 0.8, "power": 3, "begin_step": 0, "end_step": 37800, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}}, {"class_name": "PruneLowMagnitude", "config": {"name": "prune_low_magnitude_conv1d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "dtype": "float32", "layer": {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "pruning_schedule": {"class_name": "PolynomialDecay", "config": {"initial_sparsity": 0.5, "final_sparsity": 0.8, "power": 3, "begin_step": 0, "end_step": 37800, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}}, {"class_name": "PruneLowMagnitude", "config": {"name": "prune_low_magnitude_conv1d_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "dtype": "float32", "layer": {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "pruning_schedule": {"class_name": "PolynomialDecay", "config": {"initial_sparsity": 0.5, "final_sparsity": 0.8, "power": 3, "begin_step": 0, "end_step": 37800, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}}, {"class_name": "PruneLowMagnitude", "config": {"name": "prune_low_magnitude_dropout", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, "pruning_schedule": {"class_name": "PolynomialDecay", "config": {"initial_sparsity": 0.5, "final_sparsity": 0.8, "power": 3, "begin_step": 0, "end_step": 37800, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}}, {"class_name": "PruneLowMagnitude", "config": {"name": "prune_low_magnitude_flatten", "trainable": true, "dtype": "float32", "layer": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, "pruning_schedule": {"class_name": "PolynomialDecay", "config": {"initial_sparsity": 0.5, "final_sparsity": 0.8, "power": 3, "begin_step": 0, "end_step": 37800, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}}, {"class_name": "PruneLowMagnitude", "config": {"name": "prune_low_magnitude_dense", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "pruning_schedule": {"class_name": "PolynomialDecay", "config": {"initial_sparsity": 0.5, "final_sparsity": 0.8, "power": 3, "begin_step": 0, "end_step": 37800, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1250, 1]}}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": true, "label_smoothing": 0}}, "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Ö
pruning_vars
	layer
prunable_weights
mask
	threshold
pruning_step
regularization_losses
trainable_variables
	variables
	keras_api
+Í&call_and_return_all_conditional_losses
Î__call__"ç
_tf_keras_layerÍ{"class_name": "PruneLowMagnitude", "name": "prune_low_magnitude_conv1d", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "stateful": false, "config": {"name": "prune_low_magnitude_conv1d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "dtype": "float32", "layer": {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "pruning_schedule": {"class_name": "PolynomialDecay", "config": {"initial_sparsity": 0.5, "final_sparsity": 0.8, "power": 3, "begin_step": 0, "end_step": 37800, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1250, 1]}}
Ý
pruning_vars
	layer
prunable_weights
mask
	threshold
pruning_step
regularization_losses
trainable_variables
	variables
 	keras_api
+Ï&call_and_return_all_conditional_losses
Ð__call__"î
_tf_keras_layerÔ{"class_name": "PruneLowMagnitude", "name": "prune_low_magnitude_conv1d_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "stateful": false, "config": {"name": "prune_low_magnitude_conv1d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "dtype": "float32", "layer": {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "pruning_schedule": {"class_name": "PolynomialDecay", "config": {"initial_sparsity": 0.5, "final_sparsity": 0.8, "power": 3, "begin_step": 0, "end_step": 37800, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1248, 16]}}
Ý
!pruning_vars
	"layer
#prunable_weights
$mask
%	threshold
&pruning_step
'regularization_losses
(trainable_variables
)	variables
*	keras_api
+Ñ&call_and_return_all_conditional_losses
Ò__call__"î
_tf_keras_layerÔ{"class_name": "PruneLowMagnitude", "name": "prune_low_magnitude_conv1d_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "stateful": false, "config": {"name": "prune_low_magnitude_conv1d_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "dtype": "float32", "layer": {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "pruning_schedule": {"class_name": "PolynomialDecay", "config": {"initial_sparsity": 0.5, "final_sparsity": 0.8, "power": 3, "begin_step": 0, "end_step": 37800, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1245, 16]}}
ÿ
+pruning_vars
	,layer
-prunable_weights
.pruning_step
/regularization_losses
0trainable_variables
1	variables
2	keras_api
+Ó&call_and_return_all_conditional_losses
Ô__call__"©
_tf_keras_layer{"class_name": "PruneLowMagnitude", "name": "prune_low_magnitude_dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "prune_low_magnitude_dropout", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, "pruning_schedule": {"class_name": "PolynomialDecay", "config": {"initial_sparsity": 0.5, "final_sparsity": 0.8, "power": 3, "begin_step": 0, "end_step": 37800, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1241, 16]}}
î
3pruning_vars
	4layer
5prunable_weights
6pruning_step
7regularization_losses
8trainable_variables
9	variables
:	keras_api
+Õ&call_and_return_all_conditional_losses
Ö__call__"
_tf_keras_layerþ{"class_name": "PruneLowMagnitude", "name": "prune_low_magnitude_flatten", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "prune_low_magnitude_flatten", "trainable": true, "dtype": "float32", "layer": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, "pruning_schedule": {"class_name": "PolynomialDecay", "config": {"initial_sparsity": 0.5, "final_sparsity": 0.8, "power": 3, "begin_step": 0, "end_step": 37800, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1241, 16]}}
§

;pruning_vars
	<layer
=prunable_weights
>mask
?	threshold
@pruning_step
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
+×&call_and_return_all_conditional_losses
Ø__call__"¸
_tf_keras_layer{"class_name": "PruneLowMagnitude", "name": "prune_low_magnitude_dense", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "prune_low_magnitude_dense", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "pruning_schedule": {"class_name": "PolynomialDecay", "config": {"initial_sparsity": 0.5, "final_sparsity": 0.8, "power": 3, "begin_step": 0, "end_step": 37800, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 19856]}}
ó
Eiter

Fbeta_1

Gbeta_2
	Hdecay
Ilearning_rateJmºKm»Lm¼Mm½Nm¾Om¿PmÀQmÁJvÂKvÃLvÄMvÅNvÆOvÇPvÈQvÉ"
	optimizer
 "
trackable_list_wrapper
X
J0
K1
L2
M3
N4
O5
P6
Q7"
trackable_list_wrapper
Æ
J0
K1
2
3
4
L5
M6
7
8
9
N10
O11
$12
%13
&14
.15
616
P17
Q18
>19
?20
@21"
trackable_list_wrapper
Î
regularization_losses
	trainable_variables
Rmetrics
Slayer_regularization_losses
Tnon_trainable_variables

Ulayers
Vlayer_metrics

	variables
Ë__call__
Ì_default_save_signature
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses"
_generic_user_object
-
Ùserving_default"
signature_map
'
W0"
trackable_list_wrapper
±


Jkernel
Kbias
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
+Ú&call_and_return_all_conditional_losses
Û__call__"	
_tf_keras_layerð{"class_name": "Conv1D", "name": "conv1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "stateful": false, "config": {"name": "conv1d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1250, 1]}}
'
J0"
trackable_list_wrapper
5:3(2prune_low_magnitude_conv1d/mask
.:, (2$prune_low_magnitude_conv1d/threshold
/:-	 2'prune_low_magnitude_conv1d/pruning_step
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
C
J0
K1
2
3
4"
trackable_list_wrapper
°
regularization_losses
trainable_variables
\layer_regularization_losses
]metrics
^non_trainable_variables

_layers
`layer_metrics
	variables
Î__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses"
_generic_user_object
'
a0"
trackable_list_wrapper
·


Lkernel
Mbias
bregularization_losses
ctrainable_variables
d	variables
e	keras_api
+Ü&call_and_return_all_conditional_losses
Ý__call__"	
_tf_keras_layerö{"class_name": "Conv1D", "name": "conv1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "stateful": false, "config": {"name": "conv1d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1248, 16]}}
'
L0"
trackable_list_wrapper
7:5(2!prune_low_magnitude_conv1d_1/mask
0:. (2&prune_low_magnitude_conv1d_1/threshold
1:/	 2)prune_low_magnitude_conv1d_1/pruning_step
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
C
L0
M1
2
3
4"
trackable_list_wrapper
°
regularization_losses
trainable_variables
flayer_regularization_losses
gmetrics
hnon_trainable_variables

ilayers
jlayer_metrics
	variables
Ð__call__
+Ï&call_and_return_all_conditional_losses
'Ï"call_and_return_conditional_losses"
_generic_user_object
'
k0"
trackable_list_wrapper
·


Nkernel
Obias
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
+Þ&call_and_return_all_conditional_losses
ß__call__"	
_tf_keras_layerö{"class_name": "Conv1D", "name": "conv1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "stateful": false, "config": {"name": "conv1d_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1245, 16]}}
'
N0"
trackable_list_wrapper
7:5(2!prune_low_magnitude_conv1d_2/mask
0:. (2&prune_low_magnitude_conv1d_2/threshold
1:/	 2)prune_low_magnitude_conv1d_2/pruning_step
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
C
N0
O1
$2
%3
&4"
trackable_list_wrapper
°
'regularization_losses
(trainable_variables
player_regularization_losses
qmetrics
rnon_trainable_variables

slayers
tlayer_metrics
)	variables
Ò__call__
+Ñ&call_and_return_all_conditional_losses
'Ñ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
Á
uregularization_losses
vtrainable_variables
w	variables
x	keras_api
+à&call_and_return_all_conditional_losses
á__call__"°
_tf_keras_layer{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
 "
trackable_list_wrapper
0:.	 2(prune_low_magnitude_dropout/pruning_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
.0"
trackable_list_wrapper
°
/regularization_losses
0trainable_variables
ylayer_regularization_losses
zmetrics
{non_trainable_variables

|layers
}layer_metrics
1	variables
Ô__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
Ã
~regularization_losses
trainable_variables
	variables
	keras_api
+â&call_and_return_all_conditional_losses
ã__call__"°
_tf_keras_layer{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
 "
trackable_list_wrapper
0:.	 2(prune_low_magnitude_flatten/pruning_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
60"
trackable_list_wrapper
µ
7regularization_losses
8trainable_variables
 layer_regularization_losses
metrics
non_trainable_variables
layers
layer_metrics
9	variables
Ö__call__
+Õ&call_and_return_all_conditional_losses
'Õ"call_and_return_conditional_losses"
_generic_user_object
(
0"
trackable_list_wrapper
×

Pkernel
Qbias
regularization_losses
trainable_variables
	variables
	keras_api
+ä&call_and_return_all_conditional_losses
å__call__"¬
_tf_keras_layer{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 19856}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 19856]}}
'
P0"
trackable_list_wrapper
2:0
(2prune_low_magnitude_dense/mask
-:+ (2#prune_low_magnitude_dense/threshold
.:,	 2&prune_low_magnitude_dense/pruning_step
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
C
P0
Q1
>2
?3
@4"
trackable_list_wrapper
µ
Aregularization_losses
Btrainable_variables
 layer_regularization_losses
metrics
non_trainable_variables
layers
layer_metrics
C	variables
Ø__call__
+×&call_and_return_all_conditional_losses
'×"call_and_return_conditional_losses"
_generic_user_object
:	 (2iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
#:!2conv1d/kernel
:2conv1d/bias
%:#2conv1d_1/kernel
:2conv1d_1/bias
%:#2conv1d_2/kernel
:2conv1d_2/bias
 :
2dense/kernel
:2
dense/bias
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper

0
1
2
3
4
5
$6
%7
&8
.9
610
>11
?12
@13"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_dict_wrapper
6
J0
1
2"
trackable_tuple_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
µ
Xregularization_losses
Ytrainable_variables
 layer_regularization_losses
metrics
non_trainable_variables
layers
layer_metrics
Z	variables
Û__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
6
L0
1
2"
trackable_tuple_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
µ
bregularization_losses
ctrainable_variables
 layer_regularization_losses
metrics
non_trainable_variables
layers
layer_metrics
d	variables
Ý__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
6
N0
$1
%2"
trackable_tuple_wrapper
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
µ
lregularization_losses
mtrainable_variables
 layer_regularization_losses
metrics
non_trainable_variables
 layers
¡layer_metrics
n	variables
ß__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
'
"0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
uregularization_losses
vtrainable_variables
 ¢layer_regularization_losses
£metrics
¤non_trainable_variables
¥layers
¦layer_metrics
w	variables
á__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
.0"
trackable_list_wrapper
'
,0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¶
~regularization_losses
trainable_variables
 §layer_regularization_losses
¨metrics
©non_trainable_variables
ªlayers
«layer_metrics
	variables
ã__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
60"
trackable_list_wrapper
'
40"
trackable_list_wrapper
 "
trackable_dict_wrapper
6
P0
>1
?2"
trackable_tuple_wrapper
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
¸
regularization_losses
trainable_variables
 ¬layer_regularization_losses
­metrics
®non_trainable_variables
¯layers
°layer_metrics
	variables
å__call__
+ä&call_and_return_all_conditional_losses
'ä"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
>0
?1
@2"
trackable_list_wrapper
'
<0"
trackable_list_wrapper
 "
trackable_dict_wrapper
¿

±total

²count
³	variables
´	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
ÿ

µtotal

¶count
·
_fn_kwargs
¸	variables
¹	keras_api"³
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
±0
²1"
trackable_list_wrapper
.
³	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
µ0
¶1"
trackable_list_wrapper
.
¸	variables"
_generic_user_object
(:&2Adam/conv1d/kernel/m
:2Adam/conv1d/bias/m
*:(2Adam/conv1d_1/kernel/m
 :2Adam/conv1d_1/bias/m
*:(2Adam/conv1d_2/kernel/m
 :2Adam/conv1d_2/bias/m
%:#
2Adam/dense/kernel/m
:2Adam/dense/bias/m
(:&2Adam/conv1d/kernel/v
:2Adam/conv1d/bias/v
*:(2Adam/conv1d_1/kernel/v
 :2Adam/conv1d_1/bias/v
*:(2Adam/conv1d_2/kernel/v
 :2Adam/conv1d_2/bias/v
%:#
2Adam/dense/kernel/v
:2Adam/dense/bias/v
æ2ã
F__inference_sequential_layer_call_and_return_conditional_losses_317164
F__inference_sequential_layer_call_and_return_conditional_losses_317198
F__inference_sequential_layer_call_and_return_conditional_losses_318307
F__inference_sequential_layer_call_and_return_conditional_losses_318241À
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
ú2÷
+__inference_sequential_layer_call_fn_317302
+__inference_sequential_layer_call_fn_317365
+__inference_sequential_layer_call_fn_318356
+__inference_sequential_layer_call_fn_318385À
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
é2æ
!__inference__wrapped_model_315945À
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *0¢-
+(
conv1d_inputÿÿÿÿÿÿÿÿÿâ	
ð2í
V__inference_prune_low_magnitude_conv1d_layer_call_and_return_conditional_losses_318555
V__inference_prune_low_magnitude_conv1d_layer_call_and_return_conditional_losses_318575º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
º2·
;__inference_prune_low_magnitude_conv1d_layer_call_fn_318590
;__inference_prune_low_magnitude_conv1d_layer_call_fn_318601º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ô2ñ
X__inference_prune_low_magnitude_conv1d_1_layer_call_and_return_conditional_losses_318771
X__inference_prune_low_magnitude_conv1d_1_layer_call_and_return_conditional_losses_318791º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¾2»
=__inference_prune_low_magnitude_conv1d_1_layer_call_fn_318806
=__inference_prune_low_magnitude_conv1d_1_layer_call_fn_318817º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ô2ñ
X__inference_prune_low_magnitude_conv1d_2_layer_call_and_return_conditional_losses_319007
X__inference_prune_low_magnitude_conv1d_2_layer_call_and_return_conditional_losses_318987º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¾2»
=__inference_prune_low_magnitude_conv1d_2_layer_call_fn_319022
=__inference_prune_low_magnitude_conv1d_2_layer_call_fn_319033º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ò2ï
W__inference_prune_low_magnitude_dropout_layer_call_and_return_conditional_losses_319127
W__inference_prune_low_magnitude_dropout_layer_call_and_return_conditional_losses_319132º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¼2¹
<__inference_prune_low_magnitude_dropout_layer_call_fn_319144
<__inference_prune_low_magnitude_dropout_layer_call_fn_319139º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ò2ï
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_319232
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_319238º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¼2¹
<__inference_prune_low_magnitude_flatten_layer_call_fn_319250
<__inference_prune_low_magnitude_flatten_layer_call_fn_319245º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_319415
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_319430º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¸2µ
:__inference_prune_low_magnitude_dense_layer_call_fn_319445
:__inference_prune_low_magnitude_dense_layer_call_fn_319456º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
8B6
$__inference_signature_wrapper_317404conv1d_input
2
B__inference_conv1d_layer_call_and_return_conditional_losses_315962Ê
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
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ù2ö
'__inference_conv1d_layer_call_fn_315972Ê
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
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
D__inference_conv1d_1_layer_call_and_return_conditional_losses_315989Ê
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
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
û2ø
)__inference_conv1d_1_layer_call_fn_315999Ê
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
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
D__inference_conv1d_2_layer_call_and_return_conditional_losses_316016Ê
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
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
û2ø
)__inference_conv1d_2_layer_call_fn_316026Ê
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
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
º2·´
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
º2·´
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
¨2¥¢
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
¨2¥¢
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
¨2¥¢
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
¨2¥¢
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
 Ç
!__inference__wrapped_model_315945¡JKLMN$OP>Q:¢7
0¢-
+(
conv1d_inputÿÿÿÿÿÿÿÿÿâ	
ª "UªR
P
prune_low_magnitude_dense30
prune_low_magnitude_denseÿÿÿÿÿÿÿÿÿ¾
D__inference_conv1d_1_layer_call_and_return_conditional_losses_315989vLM<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
)__inference_conv1d_1_layer_call_fn_315999iLM<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¾
D__inference_conv1d_2_layer_call_and_return_conditional_losses_316016vNO<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
)__inference_conv1d_2_layer_call_fn_316026iNO<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¼
B__inference_conv1d_layer_call_and_return_conditional_losses_315962vJK<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
'__inference_conv1d_layer_call_fn_315972iJK<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÉ
X__inference_prune_low_magnitude_conv1d_1_layer_call_and_return_conditional_losses_318771mLM8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿà	
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÝ	
 Ç
X__inference_prune_low_magnitude_conv1d_1_layer_call_and_return_conditional_losses_318791kLM8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿà	
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÝ	
 ¡
=__inference_prune_low_magnitude_conv1d_1_layer_call_fn_318806`LM8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿà	
p
ª "ÿÿÿÿÿÿÿÿÿÝ	
=__inference_prune_low_magnitude_conv1d_1_layer_call_fn_318817^LM8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿà	
p 
ª "ÿÿÿÿÿÿÿÿÿÝ	É
X__inference_prune_low_magnitude_conv1d_2_layer_call_and_return_conditional_losses_318987m&N$%O8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÝ	
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÙ	
 Ç
X__inference_prune_low_magnitude_conv1d_2_layer_call_and_return_conditional_losses_319007kN$O8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÝ	
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÙ	
 ¡
=__inference_prune_low_magnitude_conv1d_2_layer_call_fn_319022`&N$%O8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÝ	
p
ª "ÿÿÿÿÿÿÿÿÿÙ	
=__inference_prune_low_magnitude_conv1d_2_layer_call_fn_319033^N$O8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÝ	
p 
ª "ÿÿÿÿÿÿÿÿÿÙ	Ç
V__inference_prune_low_magnitude_conv1d_layer_call_and_return_conditional_losses_318555mJK8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿâ	
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿà	
 Å
V__inference_prune_low_magnitude_conv1d_layer_call_and_return_conditional_losses_318575kJK8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿâ	
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿà	
 
;__inference_prune_low_magnitude_conv1d_layer_call_fn_318590`JK8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿâ	
p
ª "ÿÿÿÿÿÿÿÿÿà	
;__inference_prune_low_magnitude_conv1d_layer_call_fn_318601^JK8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿâ	
p 
ª "ÿÿÿÿÿÿÿÿÿà	¾
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_319415e@P>?Q5¢2
+¢(
"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_319430cP>Q5¢2
+¢(
"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
:__inference_prune_low_magnitude_dense_layer_call_fn_319445X@P>?Q5¢2
+¢(
"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
:__inference_prune_low_magnitude_dense_layer_call_fn_319456VP>Q5¢2
+¢(
"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿÄ
W__inference_prune_low_magnitude_dropout_layer_call_and_return_conditional_losses_319127i.8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÙ	
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÙ	
 Á
W__inference_prune_low_magnitude_dropout_layer_call_and_return_conditional_losses_319132f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÙ	
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÙ	
 
<__inference_prune_low_magnitude_dropout_layer_call_fn_319139\.8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÙ	
p
ª "ÿÿÿÿÿÿÿÿÿÙ	
<__inference_prune_low_magnitude_dropout_layer_call_fn_319144Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÙ	
p 
ª "ÿÿÿÿÿÿÿÿÿÙ	Á
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_319232f68¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÙ	
p
ª "'¢$

0ÿÿÿÿÿÿÿÿÿ
 ¾
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_319238c8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÙ	
p 
ª "'¢$

0ÿÿÿÿÿÿÿÿÿ
 
<__inference_prune_low_magnitude_flatten_layer_call_fn_319245Y68¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÙ	
p
ª "ÿÿÿÿÿÿÿÿÿ
<__inference_prune_low_magnitude_flatten_layer_call_fn_319250V8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÙ	
p 
ª "ÿÿÿÿÿÿÿÿÿÎ
F__inference_sequential_layer_call_and_return_conditional_losses_317164JKLM&N$%O.6@P>?QB¢?
8¢5
+(
conv1d_inputÿÿÿÿÿÿÿÿÿâ	
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ã
F__inference_sequential_layer_call_and_return_conditional_losses_317198yJKLMN$OP>QB¢?
8¢5
+(
conv1d_inputÿÿÿÿÿÿÿÿÿâ	
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ç
F__inference_sequential_layer_call_and_return_conditional_losses_318241}JKLM&N$%O.6@P>?Q<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿâ	
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ½
F__inference_sequential_layer_call_and_return_conditional_losses_318307sJKLMN$OP>Q<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿâ	
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¥
+__inference_sequential_layer_call_fn_317302vJKLM&N$%O.6@P>?QB¢?
8¢5
+(
conv1d_inputÿÿÿÿÿÿÿÿÿâ	
p

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_317365lJKLMN$OP>QB¢?
8¢5
+(
conv1d_inputÿÿÿÿÿÿÿÿÿâ	
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_318356pJKLM&N$%O.6@P>?Q<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿâ	
p

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_318385fJKLMN$OP>Q<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿâ	
p 

 
ª "ÿÿÿÿÿÿÿÿÿÚ
$__inference_signature_wrapper_317404±JKLMN$OP>QJ¢G
¢ 
@ª=
;
conv1d_input+(
conv1d_inputÿÿÿÿÿÿÿÿÿâ	"UªR
P
prune_low_magnitude_dense30
prune_low_magnitude_denseÿÿÿÿÿÿÿÿÿ