use kappendlist::AppendList;

fn main() {
    let size = 100;
    let l = AppendList::new();
    let mut refs: Vec<&usize> = Vec::new();

    for i in 0..size {
        assert_eq!(l.len(), i);

        refs.push(l.push(i));
        // refs.push(&l[i]);

        assert_eq!(l.len(), i + 1);
    }

    for i in 0..size {
        assert_eq!(Some(refs[i]), l.get(i));
        assert_eq!(Some(refs[i] as *const _), l.get(i).map(|x| x as *const _));
    }
    let mut l = l;
    for (i, x) in l.drain_all().enumerate() {
        assert_eq!(x, i);
        // Cannot use here since l is being borrowed.
        // If you comment this out, then it's fine.
        // assert_eq!(x, *refs[i]);
    }
    assert_eq!(l.len(), 0);
    assert!(l.is_empty());
    l.push(1);
    assert_eq!(*refs[0], 0);
}
