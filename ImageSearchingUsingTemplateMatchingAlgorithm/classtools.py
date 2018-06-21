class AttrDisplay:

    def __gatherAttrs(self):
        attrs = []
        for key in sorted(self.__dict__):
            attrs.append( '%s => %s' % (key, getattr(self, key)) )
        return ' , \n'.join(attrs)

    def __repr__(self):
        return '[%s : %s]' % (self.__class__.__name__, self.__gatherAttrs())
