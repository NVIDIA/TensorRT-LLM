Context:
disaggregated serving with partial block reuse enabled and beam width > 1

Bug symptom:
First request returns a good response but further requests return incomprehensible response with repeated tokens

There is no bug if any of the conditions (disaggregated serving with partial block reuse enabled and beam width > 1) are not met, this combination is required

Modify examples/disaggregated/simpler_example/run.sh to run 
Dump the kv cache 