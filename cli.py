import click
from src import system_data as sysdat
from src.system_data import Ad, Bd, Cd, Dd
from src.project import multiple_projects
from src.utils import save, save_json
import cvxpy as cvx


@click.group()
def cli():
    pass


@cli.command()
@click.option("-o", "--out-path", type=click.Path(exists=False), required=True)
@click.option("-r", "--radius", type=float, required=True)
def project():
    pass


@cli.command()
@click.option("-o", "--out-path", type=click.Path(exists=False), required=True)
@click.option("-r", "--radius-step", type=float, required=True)
@click.option("-c", "--center-step", type=float, required=True)
def multiproject(out_path, radius_step, center_step):
    controllers = multiple_projects(
        Ad,
        Bd,
        Cd,
        Dd,
        sysdat.csys,
        sysdat.Ts,
        q_step=center_step,
        r_step=radius_step,
        solver=cvx.SCS,
    )
    save_json([c.todict() for c in controllers], out_path)


if __name__ == "__main__":

    cli()
    # import system_data as sysdat
    # from system_data import Ad, Bd, Cd, Dd

    # save("data/controllers_solver-SCS_all.pkl", controllers)
#    eval_controllers('data/controllers_solver-SCS_all.pkl', sysdat.csys, sysdat.Ts, controllers)


#    controllers = multiple_project(Ad, Bd, Cd, Dd, solver=cvx.MOSEK)
#    eval_controllers('data/controllers_solver-MOSEK_all.pkl', sysdat.csys, sysdat.Ts, controllers)
