import skfuzzy as fuzz
import skfuzzy.control as ctrl
import numpy as np


distance  = ctrl.Antecedent(np.arange(0,101,1), 'distance')
speed  = ctrl.Antecedent(np.arange(0,101,1), 'speed')
braking_power  = ctrl.Consequent(np.arange(0,101,1), 'braking_power')


#Membership functions for distance
distance['small'] = fuzz.trimf(distance.universe, [0, 0, 25])
distance['medium'] = fuzz.trimf(distance.universe, [15, 35, 55])
distance['high'] = fuzz.trimf(distance.universe, [45, 65, 85])
distance['very_high'] = fuzz.trimf(distance.universe, [75, 100, 100])

#Membership functions for speed
speed['small'] = fuzz.trimf(speed.universe, [0, 0, 25])
speed['medium'] = fuzz.trimf(speed.universe, [15, 35, 55])
speed['high'] = fuzz.trimf(speed.universe, [45, 65, 85])
speed['very_high'] = fuzz.trimf(speed.universe, [75, 100, 100])

#Membership functions for braking power
braking_power['small'] = fuzz.trimf(braking_power.universe, [0, 0, 25])
braking_power['medium'] = fuzz.trimf(braking_power.universe, [15, 35, 55])
braking_power['high'] = fuzz.trimf(braking_power.universe, [45, 65, 85])
braking_power['very_high'] = fuzz.trimf(braking_power.universe, [75, 100, 100])

# Fuzzy rule base (4x4 FAM table)
rule1 = ctrl.Rule(distance['small'] & speed['small'], braking_power['medium'])
rule2 = ctrl.Rule(distance['small'] & speed['medium'], braking_power['high'])
rule3 = ctrl.Rule(distance['small'] & speed['high'], braking_power['very_high'])
rule4 = ctrl.Rule(distance['small'] & speed['very_high'], braking_power['very_high'])

rule5 = ctrl.Rule(distance['medium'] & speed['small'], braking_power['small'])
rule6 = ctrl.Rule(distance['medium'] & speed['medium'], braking_power['medium'])
rule7 = ctrl.Rule(distance['medium'] & speed['high'], braking_power['high'])
rule8 = ctrl.Rule(distance['medium'] & speed['very_high'], braking_power['very_high'])

rule9 = ctrl.Rule(distance['high'] & speed['small'], braking_power['small'])
rule10 = ctrl.Rule(distance['high'] & speed['medium'], braking_power['small'])
rule11 = ctrl.Rule(distance['high'] & speed['high'], braking_power['medium'])
rule12 = ctrl.Rule(distance['high'] & speed['very_high'], braking_power['high'])

rule13 = ctrl.Rule(distance['very_high'] & speed['small'], braking_power['small'])
rule14 = ctrl.Rule(distance['very_high'] & speed['medium'], braking_power['small'])
rule15 = ctrl.Rule(distance['very_high'] & speed['high'], braking_power['medium'])
rule16 = ctrl.Rule(distance['very_high'] & speed['very_high'], braking_power['high'])

braking_ctrl=ctrl.ControlSystem([rule1, rule2, rule3, rule4,rule5, rule6, rule7,rule9, rule10, rule11, rule12,rule13, rule14, rule15, rule16])
braking = ctrl.ControlSystemSimulation(braking_ctrl)
# Input values
braking.input['distance'] = 30
braking.input['speed'] = 70
braking.compute()
print(f"Braking Power: {braking.output['braking_power']}")
braking_power.view(sim=braking)