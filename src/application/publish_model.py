from ..domain.entities.configuration import ProjectConfig
from ..domain.interfaces.publisher import IModelPublisher


class PublishModelUseCase:
    """Caso de uso para publicar o modelo treinado."""

    def __init__(self, publisher: IModelPublisher):
        self.publisher = publisher

    def execute(self, config: ProjectConfig) -> None:
        """
        Executa a publicação do modelo.

        Args:
            config: Configuração do projeto.
        """
        self.publisher.publish(config)
