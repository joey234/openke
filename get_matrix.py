import json
import numpy as np
import config
import models

con = config.Config()
con.set_in_path("./OpenKE_Data/")
# con.set_test_link_prediction(True)
con.set_test_triple_classification(True)
con.set_work_threads(8)
con.set_dimension(100)
con.init()
con.set_model(models.TransE)
con.import_variables("./trained_model/model.vec.tf")
con.test()
# con.predict_triple(1365, 216, 0, thresh = 0.8)

#4433	3044	0

#1365	216	0