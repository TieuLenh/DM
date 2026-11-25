def findDecision(obj): #obj[0]: Weather, obj[1]: Temperature, obj[2]: Humidity, obj[3]: Wind
   # {"feature": "Weather", "instances": 375, "metric_value": 0.9937, "depth": 1}
   if obj[0] == 'Sunny':
      # {"feature": "Humidity", "instances": 134, "metric_value": 0.9208, "depth": 2}
      if obj[2] == 'High':
         # {"feature": "Wind", "instances": 68, "metric_value": 0.3228, "depth": 3}
         if obj[3] == 'Weak':
            # {"feature": "Temperature", "instances": 48, "metric_value": 0.1461, "depth": 4}
            if obj[1] == 'Mild':
               return 'No'
            elif obj[1] == 'Hot':
               return 'No'
            elif obj[1] == 'Cool':
               return 'No'
            else:
               return 'No'
         elif obj[3] == 'Strong':
            # {"feature": "Temperature", "instances": 20, "metric_value": 0.6098, "depth": 4}
            if obj[1] == 'Hot':
               return 'No'
            elif obj[1] == 'Mild':
               return 'No'
            elif obj[1] == 'Cool':
               return 'No'
            else:
               return 'No'
         else:
            return 'No'
      elif obj[2] == 'Normal':
         # {"feature": "Temperature", "instances": 66, "metric_value": 0.9572, "depth": 3}
         if obj[1] == 'Mild':
            # {"feature": "Wind", "instances": 28, "metric_value": 0.9852, "depth": 4}
            if obj[3] == 'Weak':
               return 'No'
            elif obj[3] == 'Strong':
               return 'No'
            else:
               return 'No'
         elif obj[1] == 'Hot':
            # {"feature": "Wind", "instances": 23, "metric_value": 0.9321, "depth": 4}
            if obj[3] == 'Weak':
               return 'Yes'
            elif obj[3] == 'Strong':
               return 'Yes'
            else:
               return 'Yes'
         elif obj[1] == 'Cool':
            # {"feature": "Wind", "instances": 15, "metric_value": 0.3534, "depth": 4}
            if obj[3] == 'Weak':
               return 'Yes'
            elif obj[3] == 'Strong':
               return 'Yes'
            else:
               return 'Yes'
         else:
            return 'Yes'
      else:
         return 'No'
   elif obj[0] == 'Cloudy':
      # {"feature": "Temperature", "instances": 127, "metric_value": 0.2394, "depth": 2}
      if obj[1] == 'Mild':
         return 'Yes'
      elif obj[1] == 'Hot':
         # {"feature": "Wind", "instances": 43, "metric_value": 0.4465, "depth": 3}
         if obj[3] == 'Strong':
            # {"feature": "Humidity", "instances": 22, "metric_value": 0.2668, "depth": 4}
            if obj[2] == 'Normal':
               return 'Yes'
            elif obj[2] == 'High':
               return 'Yes'
            else:
               return 'Yes'
         elif obj[3] == 'Weak':
            # {"feature": "Humidity", "instances": 21, "metric_value": 0.5917, "depth": 4}
            if obj[2] == 'High':
               return 'Yes'
            elif obj[2] == 'Normal':
               return 'Yes'
            else:
               return 'Yes'
         else:
            return 'Yes'
      elif obj[1] == 'Cool':
         # {"feature": "Humidity", "instances": 30, "metric_value": 0.2108, "depth": 3}
         if obj[2] == 'High':
            return 'Yes'
         elif obj[2] == 'Normal':
            # {"feature": "Wind", "instances": 15, "metric_value": 0.3534, "depth": 4}
            if obj[3] == 'Weak':
               return 'Yes'
            elif obj[3] == 'Strong':
               return 'Yes'
            else:
               return 'Yes'
         else:
            return 'Yes'
      else:
         return 'Yes'
   elif obj[0] == 'Rainy':
      # {"feature": "Temperature", "instances": 114, "metric_value": 0.1756, "depth": 2}
      if obj[1] == 'Mild':
         # {"feature": "Humidity", "instances": 43, "metric_value": 0.1594, "depth": 3}
         if obj[2] == 'Normal':
            # {"feature": "Wind", "instances": 28, "metric_value": 0.2223, "depth": 4}
            if obj[3] == 'Weak':
               return 'No'
            elif obj[3] == 'Strong':
               return 'No'
            else:
               return 'No'
         elif obj[2] == 'High':
            return 'No'
         else:
            return 'No'
      elif obj[1] == 'Hot':
         return 'No'
      elif obj[1] == 'Cool':
         # {"feature": "Humidity", "instances": 31, "metric_value": 0.3451, "depth": 3}
         if obj[2] == 'High':
            # {"feature": "Wind", "instances": 21, "metric_value": 0.4537, "depth": 4}
            if obj[3] == 'Weak':
               return 'No'
            elif obj[3] == 'Strong':
               return 'No'
            else:
               return 'No'
         elif obj[2] == 'Normal':
            return 'No'
         else:
            return 'No'
      else:
         return 'No'
   else:
      return 'No'
