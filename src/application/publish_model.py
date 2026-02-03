from ..domain.entities.configuration import ProjectConfig
from ..domain.interfaces.publisher import IModelPublisher


class PublishModelUseCase:
    """Use case to publish the trained model."""

    def __init__(self, publisher: IModelPublisher):
        self.publisher = publisher

    def execute(self, config: ProjectConfig) -> None:
        """
        Executes the model publication.

        Args:
            config: Project configuration.
        """
        self.publisher.publish(config)
