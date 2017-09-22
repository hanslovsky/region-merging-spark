package de.hanslovsky.regionmerging;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.function.IntFunction;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;

import de.hanslovsky.graph.UndirectedGraph;
import de.hanslovsky.graph.edge.Edge;
import de.hanslovsky.graph.edge.EdgeMerger;
import de.hanslovsky.graph.edge.EdgeWeight;
import de.hanslovsky.util.unionfind.HashMapStoreUnionFind;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TLongArrayList;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TLongIntHashMap;
import gnu.trove.map.hash.TLongLongHashMap;
import gnu.trove.map.hash.TLongObjectHashMap;
import gnu.trove.set.hash.TIntHashSet;
import net.imglib2.util.Pair;
import scala.Tuple2;
import scala.Tuple3;

public class BlockedRegionMergingSpark
{

	public static class Data
	{

		private final TDoubleArrayList edges;

		private final HashMap< HashWrapper< long[] >, TIntHashSet > nonContractingEdges;

		private final TLongLongHashMap counts;

		public Data( final TDoubleArrayList edges, final HashMap< HashWrapper< long[] >, TIntHashSet > nonContractingEdges, final TLongLongHashMap counts )
		{
			super();
			this.edges = edges;
			this.nonContractingEdges = nonContractingEdges;
			this.counts = counts;
		}

		public TDoubleArrayList edges()
		{
			return edges;
		}

		public HashMap< HashWrapper< long[] >, TIntHashSet > nonContractingEdges()
		{
			return nonContractingEdges;
		}

		public TLongLongHashMap counts()
		{
			return counts;
		}

	}

	public static class Options
	{
		private double threshold;

		private StorageLevel persistenceLevel;

		private final int minimuMultiplicity = 0;

		public static Options options()
		{
			return new Options( 1.0, StorageLevel.MEMORY_ONLY() );
		}

		public Options( final double threshold, final StorageLevel persistenceLevel )
		{
			super();
			this.threshold = threshold;
			this.persistenceLevel = persistenceLevel;
		}

		public Options copy()
		{
			return new Options( threshold, persistenceLevel.clone() );
		}

		public Options threshold( final double threshold )
		{
			this.threshold = threshold;
			return this;
		}

		public Options persistenceLevel( final StorageLevel persistenceLevel )
		{
			this.persistenceLevel = persistenceLevel;
			return this;
		}

		public StorageLevel persistenceLevel()
		{
			return this.persistenceLevel.clone();
		}

		public double threshold()
		{
			return this.threshold;
		}
	}

	public static interface MergeHandlerFactory
	{

		public MergeNotify create( int iteration );

	}

	private final EdgeMerger merger;

	private final EdgeWeight edgeWeight;

	private final IntFunction< MergeNotifyWithFinishNotification > mergeNotifyGenerator;

	private final int factor;

	public BlockedRegionMergingSpark( final EdgeMerger merger, final EdgeWeight edgeWeight, final IntFunction< MergeNotifyWithFinishNotification > mergeHandlerGenerator )
	{
		this( merger, edgeWeight, mergeHandlerGenerator, 2 );
	}

	public BlockedRegionMergingSpark( final EdgeMerger merger, final EdgeWeight edgeWeight, final IntFunction< MergeNotifyWithFinishNotification > mergeNotifyGenerator, final int factor )
	{
		super();
		this.merger = merger;
		this.edgeWeight = edgeWeight;
		this.mergeNotifyGenerator = mergeNotifyGenerator;
		this.factor = factor;
	}

	public void agglomerate(
			final JavaSparkContext sc,
			final JavaPairRDD< HashWrapper< long[] >, Data > rdd,
			final BiConsumer< Integer, JavaPairRDD< HashWrapper< long[] >, Tuple2< TLongArrayList, HashMapStoreUnionFind > > > mergesLogger,
			final Options options )
	{

		final List< Object > unpersistList = new ArrayList<>();

		final Broadcast< Options > optionsBC = sc.broadcast( options );

		JavaPairRDD< HashWrapper< long[] >, Data > targetRdd = rdd;
		int iteration = 0;
		boolean hasReachedSingleBlock = false;
		final int dataSize = merger.dataSize();
		while ( !hasReachedSingleBlock )
		{
			hasReachedSingleBlock = targetRdd.count() == 1;
			System.out.println( targetRdd.count() + " blocks at iteration " + iteration );
			final int finalIteration = iteration;
			targetRdd.map( t -> {
				System.out.println( Arrays.toString( t._1().getData() ) + ": Starting agglomeration with " + new Edge( t._2().edges(), dataSize ).size() + " edges, and " + t._2().nonContractingEdges().values().stream().mapToInt( c -> c.size() ).sum() + " non-contracting edges." );
				return t;
			} ).count();
			final JavaPairRDD< HashWrapper< long[] >, Tuple3< Data, TLongArrayList, HashMapStoreUnionFind > > mergedEdgesWithMergeLog = createGraphAndContractMinimalEdges( targetRdd, optionsBC, merger, edgeWeight, mergeNotifyGenerator, finalIteration );
			mergedEdgesWithMergeLog.cache();
			final JavaPairRDD< HashWrapper< long[] >, Data > mergedEdges = mergedEdgesWithMergeLog.mapValues( p -> p._1() );
			mergesLogger.accept( iteration, mergedEdgesWithMergeLog.mapValues( p -> new Tuple2<>( p._2(), p._3() ) ) );

			final JavaPairRDD< HashWrapper< long[] >, Data > remapped = adjustPosition( mergedEdges, this.factor );

			final JavaPairRDD< HashWrapper< long[] >, List< Data > > aggregated = aggregateAsList( remapped );

			final JavaPairRDD< HashWrapper< long[] >, Data > reduced = combineSubgraphs( aggregated, edgeWeight.dataSize() );

			final JavaPairRDD< HashWrapper< long[] >, Data > previous = targetRdd;
			targetRdd = reduced.persist( options.persistenceLevel );
			targetRdd.count();
			previous.unpersist();
			mergedEdgesWithMergeLog.unpersist();

			System.out.println( "Finished iteration " + iteration );
			++iteration;

		}

		if ( targetRdd != rdd )
			targetRdd.unpersist();

	}

	private static long[] adjustPosition( final long[] position, final long factor )
	{
		return Arrays.stream( position ).map( p -> p / factor ).toArray();
	}

	private static TLongObjectHashMap< TLongIntHashMap > createNodeEdgeMap( final Edge e )
	{
		final TLongObjectHashMap< TLongIntHashMap > nodeEdgeMap = new TLongObjectHashMap<>();
		for ( int edgeIndex = 0; edgeIndex < e.size(); ++edgeIndex )
		{
			e.setIndex( edgeIndex );
			if ( e.isValid() )
			{
				final long from = e.from();
				final long to = e.to();

				if ( !nodeEdgeMap.contains( from ) )
					nodeEdgeMap.put( from, new TLongIntHashMap() );
				if ( !nodeEdgeMap.contains( to ) )
					nodeEdgeMap.put( to, new TLongIntHashMap() );

				final TLongIntHashMap fromMap = nodeEdgeMap.get( from );
				final TLongIntHashMap toMap = nodeEdgeMap.get( to );

				if ( fromMap.contains( to ) )
				{
					assert toMap.contains( from ): "Map inconsistency!";
					e.setObsolete();
				}
				else
				{
					assert !toMap.contains( from ): "Map inconsistency!";
					fromMap.put( to, edgeIndex );
					toMap.put( from, edgeIndex );
				}
			}
		}
		return nodeEdgeMap;
	}

	private static JavaPairRDD< HashWrapper< long[] >, Data > adjustPosition( final JavaPairRDD< HashWrapper< long[] >, Data > input, final int factor )
	{
		return input.mapToPair( t -> adjustPosition( t, factor ) );
	}

	private static Tuple2< HashWrapper< long[] >, Data > adjustPosition( final Tuple2< HashWrapper< long[] >, Data > t, final int factor )
	{
		final long[] key = adjustPosition( t._1().getData(), factor );
		final HashWrapper< long[] > hashedKey = new HashWrapper<>( key, t._1().getHash(), t._1().getEquals() );
		final Data data = t._2();
		final HashMap< HashWrapper< long[] >, TIntHashSet > nonContractingEdges = new HashMap<>();
		data.nonContractingEdges.forEach( ( k, v ) -> {
			final long[] adjustedKey = adjustPosition( k.getData(), factor );
			if ( !Arrays.equals( adjustedKey, key ) )
				nonContractingEdges.put( new HashWrapper<>( adjustedKey, k.getHash(), k.getEquals() ), v );
		} );
		return new Tuple2<>( hashedKey, new Data( data.edges, nonContractingEdges, data.counts ) );
	}

	private static final JavaPairRDD< HashWrapper< long[] >, Tuple3< Data, TLongArrayList, HashMapStoreUnionFind > > createGraphAndContractMinimalEdges(
			final JavaPairRDD< HashWrapper< long[] >, Data > input,
			final Broadcast< Options > optionsBC,
			final EdgeMerger merger,
			final EdgeWeight edgeWeight,
			final IntFunction< MergeNotifyWithFinishNotification > notifyGenerator,
			final int iteration )
	{
		return input.mapValues( data -> createGraphAndContractMinimalEdges( data, optionsBC, merger, edgeWeight, notifyGenerator, iteration ) );
	}

	private static Tuple3< Data, TLongArrayList, HashMapStoreUnionFind > createGraphAndContractMinimalEdges( final Data data, final Broadcast< Options > optionsBC, final EdgeMerger merger, final EdgeWeight edgeWeight, final IntFunction< MergeNotifyWithFinishNotification > notifyGenerator, final int iteration )
	{
		final TIntHashSet nonContractingEdges = new TIntHashSet();
		final Options opt = optionsBC.getValue();
		data.nonContractingEdges.values().forEach( nonContractingEdges::addAll );
		final UndirectedGraph g = new UndirectedGraph( data.edges, createNodeEdgeMap( new Edge( data.edges, edgeWeight.dataSize() ) ), edgeWeight.dataSize() );
		final MergeNotifyWithFinishNotification notify = notifyGenerator.apply( iteration );
		final Pair< TLongArrayList, HashMapStoreUnionFind > mergesAndMapping = RegionMerging.mergeLocallyMinimalEdges( g, merger, edgeWeight, data.counts, opt.threshold, opt.minimuMultiplicity, nonContractingEdges, notify );
		notify.notifyDone();
		final TLongArrayList merges = mergesAndMapping.getA();
		final HashMapStoreUnionFind mapping = mergesAndMapping.getB();

		final Edge e = new Edge( data.edges, edgeWeight.dataSize() );

		for ( int i = 0; i < e.size(); ++i )
		{
			e.setIndex( i );
			if ( e.isValid() )
			{
				e.from( mapping.findRoot( e.from() ) );
				e.to( mapping.findRoot( e.to() ) );
			}
		}

		return new Tuple3<>( data, merges, mapping );
	}

	private static JavaPairRDD< HashWrapper< long[] >, Data > combineSubgraphs( final JavaPairRDD< HashWrapper< long[] >, List< Data > > input, final int dataSize )
	{
		return input.mapValues( dataList -> combineSubgraphs( dataList, dataSize ) );
	}

	private static Data combineSubgraphs( final List< Data > dataList, final int dataSize )
	{

		final TDoubleArrayList edgeStore = new TDoubleArrayList();
		final Edge e = new Edge( edgeStore, dataSize );
		final HashMap< HashWrapper< long[] >, TIntHashSet > nonContractingEdges = new HashMap<>();
		final TLongLongHashMap counts = new TLongLongHashMap();

		for ( final Data d : dataList )
		{

			counts.putAll( counts );

			final TIntObjectHashMap< HashWrapper< long[] > > inverseNonContractingEdges = new TIntObjectHashMap<>();
			d.nonContractingEdges.forEach( ( k, v ) -> v.forEach( edgeIndex -> {
				inverseNonContractingEdges.put( edgeIndex, k );
				return true;
			} ) );

			final Edge oldEdge = new Edge( d.edges, dataSize );
			for ( int i = 0; i < oldEdge.size(); ++i )
			{
				oldEdge.setIndex( i );
				if ( oldEdge.isValid() )
				{
					final int newEdgeIndex = e.size();
					e.add( oldEdge );
					if ( inverseNonContractingEdges.contains( i ) )
					{
						final HashWrapper< long[] > key = inverseNonContractingEdges.get( i );
						if ( !nonContractingEdges.containsKey( key ) )
							nonContractingEdges.put( key, new TIntHashSet() );
						nonContractingEdges.get( key ).add( newEdgeIndex );
					}
				}
			}

		}

		return new Data( edgeStore, nonContractingEdges, counts );
	}

	private static JavaPairRDD< HashWrapper< long[] >, List< Data > > aggregateAsList( final JavaPairRDD< HashWrapper< long[] >, Data > remapped )
	{
		final JavaPairRDD< HashWrapper< long[] >, List< Data > > aggregated = remapped.aggregateByKey( new ArrayList<>(),
				( list, data ) -> {
					list.add( data );
					return list;
				},
				( list1, list2 ) -> {
					final ArrayList< Data > list = new ArrayList<>();
					list.addAll( list1 );
					list.addAll( list2 );
					return list;
				} );
		return aggregated;
	}

}
