import typer


cli = typer.Typer()


@cli.command(name="welcome", help="Say welcome with the name!", context_settings={"allow_extra_args": True}, short_help="Say welcome!")
def welcome(name: str):
    typer.echo(f"welcome {name}!")


if __name__ == "__main__":
    cli()
