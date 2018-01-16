from scipy import stats 

class DictionaryList:

    def __init__(self):
        self.d = {}
        self.summarizers = {"mean": np.mean, "std": np.std}

    def add_summarizer(self, name, func):
        self.summarizers[name] = func

    def append(self, label, item):
        if label not in self.d.keys():
            self.d[label] = []
        self.d[label].append(item)

    def append_flat(self, label, item_list):
        if label not in self.d.keys():
            self.d[label] = []

        for i in item_list:
            self.d[label].append(i)

    def summary(self):
        r = {}
        for k in self.d.keys():
            r[k] = {}
            for sk in self.summarizers.keys():
                r[k][sk] = self.summarizers[sk](np.array(self.d[k]))
        return r

    def summary_df(self):
        rs = self.summary()
        df = pd.DataFrame([rs[k].values() for k in rs.keys()], columns=rs[rs.keys()[0]].keys())
        df.index = rs.keys()
        return df.sort_index()

    def get_dataframe(self):
        rf = []
        n  = len(self[self.keys()[0]])
        for k in sorted(self.keys()):
            if len(self[k])!=n:
                raise ValueError("all items in dictionary list must contain the same number of elements")
            rf.append(self[k])
        rf = np.array(rf).T
        rf = pd.DataFrame(rf, columns=sorted(self.keys()))
        return rf

    def __getitem__(self, label):
        return np.array(self.d[label])

    def keys(self):
        return self.d.keys()

    def values(self):
        return self.d.values()

    def plot(self, xseries=None, xlabel="", figsize=(15,3), **kwargs):
        if figsize!=None:
            plt.figure(figsize=figsize)
        for i,k in enumerate(self.d.keys()):
            plt.subplot(1,len(self.d.keys()),i+1)
            plt.plot(range(len(self.d[k])) if xseries is None else xseries,
                     self.d[k], **kwargs)
            plt.xlabel(xlabel)
            plt.title(k)


def plot_signal_descriptors(signal, title=""):
    import statsmodels.tsa.api as smt
    plt.subplot(131)
    plt.plot(signal)
    plt.axhline(0, color="gray")
    plt.title(title+", mean=%.4f std=%.4f"%(np.mean(signal), np.std(signal)))
    plt.subplot(132)
    plt.hist(signal, bins=30, normed=True);
    x = np.linspace(np.min(signal), np.max(signal), 100)
    gx = stats.norm(loc=np.mean(signal), scale=np.std(signal)).pdf(x)
    plt.plot(x, gx, color="black", lw=2)
    plt.title("histogram + reference gaussian")
    ax=plt.subplot(133)
    smt.graphics.plot_acf(signal, lags=40, alpha=1., ax=ax);
