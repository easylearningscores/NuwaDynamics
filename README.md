# NuwaDynamics: Discovering and Updating in Causal Spatio-Temporal Modeling

Abstract: Spatio-temporal (ST) prediction plays a pivotal role in earth sciences, such as
meteorological prediction, urban computing. Adequate high-quality data, cou-
pled with deep models capable of inference, are both indispensable and prereq-
uisite for achieving meaningful results. However, the sparsity of data and the
high costs associated with deploying sensors lead to significant data imbalances.
Models that are overly tailored and lack causal relationships further compromise
the generalizabilities of inference methods. Towards this end, we first es-
tablish a causal concept for ST predictions, named NuwaDynamics, which tar-
gets to identify causal regions in data and endow model with causal reasoning
ability in a two-stage process. Concretely, we initially leverage upstream self-
supervision to discern causal important patches, imbuing the model with gen-
eralized information and conducting informed interventions on complementary
trivial patches to extrapolate potential test distributions. This phase is referred
to as the discovery step. Advancing beyond the discovery step, we transfer the
data to downstream tasks for targeted ST objectives, aiding the model in recog-
nizing a broader potential distribution and fostering its causal perceptual capa-
bilities (denoted as Update step). Our concept aligns seamlessly with the con-
temporary backdoor adjustment mechanism in causality theory. Extensive experi-
ments on six real-world ST benchmarks showcase that models can gain outcomes
upon the integration of the NuwaDynamics concept. NuwaDynamics also can
significantly benefit a wide range of changeable ST tasks like extreme weather
and long temporal step super-resolution predictions.
