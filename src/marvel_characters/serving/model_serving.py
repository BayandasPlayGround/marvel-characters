"""Model serving module for Marvel characters."""

from datetime import timedelta

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    EndpointStateReady,
    ServedEntityInput,
)


class ModelServing:
    """Manages model serving in Databricks for Marvel characters."""

    def __init__(self, model_name: str, endpoint_name: str) -> None:
        """Initialize the Model Serving Manager.

        :param model_name: Name of the model to be served
        :param endpoint_name: Name of the serving endpoint
        """
        self.workspace = WorkspaceClient()
        self.endpoint_name = endpoint_name
        self.model_name = model_name

    def get_latest_model_version(self) -> str:
        """Retrieve the latest version of the model.

        :return: Latest version of the model as a string
        """
        client = mlflow.MlflowClient()
        latest_version = client.get_model_version_by_alias(self.model_name, alias="latest-model").version
        print(f"Latest model version: {latest_version}")
        return latest_version

    def deploy_or_update_serving_endpoint(
        self,
        version: str = "latest",
        workload_size: str = "Small",
        scale_to_zero: bool = True,
        wait: bool = True,
        timeout_minutes: int = 20,
    ) -> None:
        """Deploy or update the model serving endpoint in Databricks for Marvel characters.

        :param version: Model version to serve (default: "latest")
        :param workload_size: Size of the serving workload (default: "Small")
        :param scale_to_zero: Whether to enable scale-to-zero (default: True)
        :param wait: Block until the endpoint reports READY and raise if it doesn't
            (default: True). A config update can finish without error while the served
            model still fails to load, leaving a silently broken endpoint; waiting and
            checking readiness here surfaces that failure immediately instead.
        :param timeout_minutes: Minutes to wait for readiness before raising (default: 20)
        """
        endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list())
        entity_version = self.get_latest_model_version() if version == "latest" else version

        served_entities = [
            ServedEntityInput(
                entity_name=self.model_name,
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size,
                entity_version=entity_version,
            )
        ]

        if not endpoint_exists:
            waiter = self.workspace.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(
                    served_entities=served_entities,
                ),
            )
        else:
            waiter = self.workspace.serving_endpoints.update_config(
                name=self.endpoint_name, served_entities=served_entities
            )

        if not wait:
            return

        endpoint = waiter.result(timeout=timedelta(minutes=timeout_minutes))
        if endpoint.state is None or endpoint.state.ready != EndpointStateReady.READY:
            ready_state = endpoint.state.ready if endpoint.state else None
            raise RuntimeError(
                f"Endpoint {self.endpoint_name!r} did not become READY serving version "
                f"{entity_version} (state.ready={ready_state}). Check the endpoint's build/service "
                "logs in the Databricks UI before retrying."
            )
