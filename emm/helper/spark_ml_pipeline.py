# Copyright (c) 2023 ING Analytics Wholesale Banking
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from typing import cast

from emm.helper import spark_installed
from emm.indexing.spark_candidate_selection import SparkCandidateSelectionEstimator

if spark_installed:
    from pyspark.ml import Pipeline, PipelineModel
    from pyspark.ml.base import Estimator, Transformer
    from pyspark.sql.dataframe import DataFrame


class EMPipeline(Pipeline):
    """Wrapper for regular spark Pipeline"""

    def _fit(self, dataset: DataFrame) -> "PipelineModel":
        """Custom fit function for spark Pipeline

        Acts just like Pipeline.fit(), but CandidateSelectionEstimator is given special treatment:
        Do not perform its costly transform step during Pipeline.fit(), after CandidateSelectionEstimator.fit().
        This step is costly and not needed for the next step: the supervised model fit().
        """
        stages = self.getStages()
        for stage in stages:
            if not (isinstance(stage, (Estimator, Transformer))):
                raise TypeError("Cannot recognize a pipeline stage of type %s." % type(stage))
        idx_last_estimator = -1
        for i, stage in enumerate(stages):
            if isinstance(stage, Estimator):
                idx_last_estimator = i
        transformers = []
        for i, stage in enumerate(stages):
            if i <= idx_last_estimator:
                if isinstance(stage, Transformer):
                    transformers.append(stage)
                    dataset = stage.transform(dataset)
                elif isinstance(stage, SparkCandidateSelectionEstimator):
                    # this step is different from Pipeline.fit()
                    model = stage.fit(dataset)
                    transformers.append(model)
                    # do not transform dataset after stage.fit();
                    # this is costly and not needed for the next step: the supervised model fit().
                else:  # must be an Estimator
                    model = stage.fit(dataset)
                    transformers.append(model)
                    if i < idx_last_estimator:
                        dataset = model.transform(dataset)
            else:
                transformers.append(cast(Transformer, stage))

        return PipelineModel(transformers)
