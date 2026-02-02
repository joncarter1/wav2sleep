"""Functions for parallel computation using Ray."""

__all__ = ('parallelise',)
from typing import Any, Callable, Iterable, Optional, TypeVar, Union

import ray
from ray import ObjectRef

# Generic input/output types
X, Y = TypeVar('X'), TypeVar('Y')

from tqdm import tqdm


def hashable(obj: Any) -> bool:
    """Check an object is hashable."""
    try:
        hash(obj)
        return True
    except TypeError:
        return False


def parallelise(
    func: Callable[[X], Y],
    iterator: Iterable[X],
    max_parallel: int = 10,
    ray_config: Optional[dict[str]] = None,
    return_refs: bool = False,
    return_dict: bool = False,
    wait_for_all: bool = False,
    shutdown_ray: bool = False,
    use_tqdm: bool = False,
) -> Union[list[Y], list['ObjectRef[Y]'], dict[X, Y], dict[X, 'ObjectRef[Y]']]:
    """Parallelise function calls over an iterator using Ray.

    Args:
        func: Function to parallelise. This should take a single argument from the iterator.
            For functions with additional, fixed arguments consider using functools.partial.
        iterator: Iterable arguments to the function.
        max_parallel: Maximum number of parallel tasks.
        ray_config: Options to pass to the ray.remote decorator. By default, each parallel process
            is assigned a single CPU without any memory constraints.
        return_refs: Return object references rather than pulling objects from the Ray object store.
            This can be useful to avoid transferring all objects back to the head node when running
            a function that returns large objects on a multi-node cluster.
        return_dict: Return a mapping from the iterable to the result rather than a list of results.
            This will only work if the iterables are hashable.
        wait_for_all: Block until all remote tasks are complete before returning.
            This option is only meaningful if return_refs=True.
        shutdown_ray: Shutdown Ray after completion. This option is only meaningful if return_refs=False.
            Otherwise Ray cannot shutdown, because the objects haven't been retrieved from the object store.
        use_tqdm: Wrap iterations in a tqdm progress bar. tqdm should be installed if this is set to True.
    Returns:
        List of results, with the same ordering as the iterator, if return_refs=False, return_dict=False
    """
    # Default to running each task on a single core w/o memory constraints.
    if ray_config is None:
        ray_config = {'num_cpus': 1}
    _ray_func = ray.remote(**ray_config)(func)
    result_refs: list['ObjectRef[Y]'] = []
    # Create a tqdm wrapper function if use_tqdm=True
    if use_tqdm and tqdm is not None:
        wrapped_iterator = tqdm(enumerate(iterator))
    else:
        wrapped_iterator = enumerate(iterator)
    # Launch up to 'max_parallel' calls to 'func'
    for i, iterable in wrapped_iterator:
        if return_dict and not hashable(iterable):
            raise TypeError(f"{iterable=} was not hashable. Can't set return_dict=True.")
        # Wait till < max_parallel running before creating new tasks.
        if len(result_refs) > max_parallel:
            num_ready = i - max_parallel
            ray.wait(result_refs, num_returns=num_ready)
        # Create a new call to 'func' that runs in parallel.
        result_refs.append(_ray_func.remote(iterable))
    if return_refs:
        if wait_for_all:
            ray.wait(result_refs, num_returns=len(result_refs))
        outputs = result_refs
    else:
        outputs: list[Y] = ray.get(result_refs)
        if shutdown_ray:
            ray.shutdown()

    if return_dict:
        return {iterable: output for iterable, output in zip(iterator, outputs)}
    else:
        return outputs
